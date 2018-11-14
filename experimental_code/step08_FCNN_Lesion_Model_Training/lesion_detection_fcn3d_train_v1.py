#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import copy
import glob
import math
import time
import shutil
import json
import numpy as np
import pandas as pd
import keras
import keras.layers as kl
import keras.models
import nibabel as nib
import logging
import matplotlib.pyplot as plt
import skimage as sk
import skimage.transform
import argparse
import logging
import tensorflow as tf
import typing
import scipy
import scipy.misc
import concurrent.futures as mp

try:
    from . import imtransform as imt
except:
    import imtransform as imt


##################################################
def _split_list_by_blocks(lst, psiz) -> list:
    tret = [lst[x:x + psiz] for x in range(0, len(lst), psiz)]
    return tret


def build_model(cfg, num_conv=2, kernelSize=3, num_flt=16, ppad='same', num_subs=5,
                is_unet=True, is_debug=False, isSkipFirstConnection=False, inp_shape=None):
    """
    :type cfg: Config
    """
    if inp_shape is None:
        inp_shape = cfg.inp_shape
    dataInput = kl.Input(shape=inp_shape)
    fsiz = (kernelSize, kernelSize, kernelSize)
    psiz = (2, 2, 2)
    x = dataInput
    # -------- Encoder --------
    lstMaxPools = []
    for cc in range(num_subs):
        for ii in range(num_conv):
            x = kl.Conv3D(filters=num_flt * (2 ** cc), kernel_size=fsiz, padding=ppad, activation='relu')(x)
        lstMaxPools.append(x)
        x = kl.MaxPooling3D(pool_size=psiz)(x)
    # -------- Decoder --------
    for cc in range(num_subs):
        for ii in range(num_conv):
            x = kl.Conv3D(filters=num_flt * (2 ** (num_subs - 1 - cc)), kernel_size=fsiz, padding=ppad, activation='relu')(x)
        x = kl.UpSampling3D(size=psiz)(x)
        if is_unet:
            if isSkipFirstConnection and (cc == (num_subs - 1)):
                continue
            x = kl.concatenate([x, lstMaxPools[-1 - cc]], axis=-1)
    # final convs
    for cc in range(num_subs):
        x = kl.Conv3D(filters=num_flt, kernel_size=fsiz, padding=ppad, activation='relu')(x)
    # 1x1 Convolution: emulation of Dense layer
    if cfg.num_cls == 2:
        x = kl.Conv3D(filters=1, kernel_size=(1, 1, 1), padding='valid', activation='sigmoid')(x)
        x = kl.Reshape([cfg.inp_shape[0], cfg.inp_shape[1], cfg.inp_shape[2]])(x)
    else:
        x = kl.Conv3D(filters=cfg.num_cls, kernel_size=(1, 1, 1), padding='valid', activation='linear')(x)
        x = kl.Reshape([-1, cfg.num_cls])(x)
        x = kl.Activation('softmax')(x)
        x = kl.Reshape([inp_shape[0], inp_shape[1], inp_shape[2], cfg.num_cls])(x)
    retModel = keras.models.Model(dataInput, x)
    if is_debug:
        retModel.summary()
        fimg_model = 'model_graph_fcnn3d.png'
        keras.utils.plot_model(retModel, fimg_model, show_shapes=True)
        plt.imshow(plt.imread(fimg_model))
        plt.show()
    return retModel


##################################################
def _norm_img3d(img3d, p1 = -2400, p2 = 1200) -> np.ndarray:
    """
    :type img3d: np.ndarray
    """
    img3d = (img3d - p1) / (p2 - p1)
    img3d[img3d < 0.0] = 0.0
    img3d[img3d > 1.0] = 1.0
    img3d = (2. * img3d) - 1
    return img3d
#
# def _get_msk3d(cls3d, msk3d, ptype = np.float32):
#     class_weight = {
#         0: 1.,
#         1: 2.,
#         2: 2.,
#         3: 2.,
#         4: 2.,
#         5: 2.,
#         6: 2.
#     }



def load_data(data) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img3d = data['img']
    cls3d = data['cls']
    msk3d = data['msk']
    if isinstance(img3d, str):
        img3d = _norm_img3d(nib.load(img3d).get_data().astype(np.float32))
    if isinstance(cls3d, str):
        cls3d = nib.load(cls3d).get_data()
        # if cls3d.min() < -1:
        #     cls3d += 1024  # FIXME: wtf??? in our data... :(
        cls3d -= cls3d.min() # FIXME: wtf??? in our data... :(
        #
        cls3d[(cls3d == 1) | (cls3d == 2) | (cls3d == 3) | (cls3d == 4) | (cls3d == 5)] = 1
        cls3d[cls3d == 6]  = 2
        cls3d[cls3d == 7]  = 3
        cls3d[cls3d == 8]  = 4
        cls3d[cls3d == 9]  = 5
        cls3d[cls3d == 10] = 6
        cls3d = cls3d.astype(np.uint8)
    if isinstance(msk3d, str):
        msk3d = (nib.load(msk3d).get_data()>0)
    return img3d, cls3d, msk3d


def get_safe_bbox(bbox, shp):
    dbbox = [xx[1] - xx[0] for xx in bbox]
    bbox_safe = []
    for ii, (xx, dxx) in enumerate(zip(bbox, dbbox)):
        yy = xx
        if xx[0] < 0:
            yy[0] = 0
            yy[1] = dxx
        if xx[1] >= shp[ii]:
            yy[0] = shp[ii] - dxx - 1
            yy[1] = yy[0] + dxx
        bbox_safe.append(yy)
    return bbox_safe


def get_crop(img3d, cls3d, msk3d, cfg, cache, idx, is_randomize=True) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :type img3d: np.ndarray
    :type cls3d: np.ndarray
    :type msk3d:  np.ndarray
    :type cfg: Config
    """
    if is_randomize:
        rnd_shp_scl = np.random.uniform(*cfg.rnd_siz_scl, 3)
    else:
        rnd_shp_scl = np.array([1, 1, 1])
    rshp = (np.array(cfg.inp_shape[:3]) * rnd_shp_scl).astype(np.int32)
    rshp_d2 = rshp //2
    #
    cls_coords = cache.get_cls_coords(cls3d, msk3d, idx)
    num_cls = len(cls_coords)
    # rnd_type: '-1' -> random near lung
    rnd_type = np.random.choice(list(cls_coords.keys()))
    rnd_coords = cls_coords[rnd_type]
    num_coords = len(rnd_coords[0])
    rnd_id_coord = np.random.randint(0, num_coords)
    xyz  = [rnd_coords[xx][rnd_id_coord] for xx in range(3)]
    bbox0 = [[xx-ds, xx-ds+ss] for xx,ds,ss in zip(xyz, rshp_d2, rshp)]
    bbox = get_safe_bbox(bbox0, cls3d.shape[:3])
    #
    img3d_crop = img3d[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    cls3d_crop = cls3d[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    msk3d_crop = msk3d[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    return (img3d_crop, cls3d_crop, msk3d_crop)

class DataCache:
    def __init__(self, num_idx):
        self.num_idx = num_idx
        self.cls_coords = {}
    def get_cls_coords(self, pcls, pmsk, idx) -> dict:
        if idx not in self.cls_coords:
            logging.info('\t[**] cache-miss (idx={}), loading... cache-state: in/tot = [{}/{}]'.format(idx, len(self.cls_coords), self.num_idx))
            cls_ids = np.unique(pcls)
            cls_ids = cls_ids[cls_ids > 0]
            tmp = {xx:np.where(pcls == xx) for xx in cls_ids}
            tmp[-1] = np.where(pmsk > 0)
            self.cls_coords[idx] = tmp
            return tmp
        return self.cls_coords[idx]

def _aug_t4_symmetry(lst_imgs) -> list:
    is_flip_v = np.random.rand() > 0.5
    is_flip_h = np.random.rand() > 0.5
    rnd_rot90 = np.random.randint(0, 4)
    if is_flip_v:
        lst_imgs = [np.flipud(xx) for xx in lst_imgs]
    if is_flip_h:
        lst_imgs = [np.fliplr(xx) for xx in lst_imgs]
    if rnd_rot90 > 0:
        lst_imgs = [np.rot90(xx, rnd_rot90) for xx in lst_imgs]
    return lst_imgs

def data_generator(cfg, is_train=True, is_randomize=True):
    if is_train:
        paths_data = cfg.paths_trn
    else:
        paths_data = cfg.paths_val
    num_data = len(paths_data)
    num_img4batch = int(math.ceil(cfg.batch_size / cfg.smpl_per_img))
    cnt_inter = 0
    #cache = DataCache(num_data)
    while True:
        if num_img4batch > num_data:
            rnd_idx = np.random.randint(0, num_data, num_img4batch)
        else:
            rnd_idx = np.random.permutation(num_data)[:num_img4batch]
        datax, datay = [], []
        cnt_batch = 0
        for idx in rnd_idx:
            if cnt_batch >= cfg.batch_size:
                break
            pdata = paths_data[idx]
            cache = DataCache(num_data)
            img3d, cls3d, msk3d = load_data(pdata)
            for ss in range(cfg.smpl_per_img):
                crop_img, crop_cls, crop_msk = get_crop(img3d, cls3d, msk3d, cfg, cache, idx, is_randomize=is_randomize)
                if crop_img.shape != cfg.inp_shape[:3]:
                    crop_img = sk.transform.resize(crop_img, cfg.inp_shape[:3], order=2)
                    crop_cls = sk.transform.resize(crop_cls.astype(np.float), cfg.inp_shape[:3], order=0).astype(np.float32)
                    crop_msk = sk.transform.resize(crop_msk, cfg.inp_shape[:3], order=1) > 0.5
                if is_randomize:
                    crop_img = np.random.uniform(*cfg.rnd_val_scl) * crop_img + np.random.uniform(*cfg.rnd_val_sht)
                yy = keras.utils.to_categorical(crop_cls.reshape(-1), num_classes=cfg.num_cls)\
                        .reshape(list(crop_img.shape[:3]) + [cfg.num_cls])
                xx = crop_img
                if is_randomize:
                    xx, yy = _aug_t4_symmetry([xx, yy])
                if cfg.is_zerobg:
                    yy[crop_msk > 0, 0] = 0
                if cfg.class_weights is not None:
                    for cci, ccw in enumerate(cfg.class_weights):
                        yy[:, :, cci] *= ccw
                xx = np.expand_dims(xx, axis=-1)
                datax.append(xx)
                datay.append(yy)
                cnt_batch += 1
        datax = np.stack(datax)
        datay = np.stack(datay)
        yield datax, datay
        cnt_inter += 1

##################################################
class Config:
    modes_run = ['train', 'val', 'test', 'infer']
    modes_cls = ['mclass', 'bclass', 'dice']
    def __init__(self, args):
        self.mode_run   = args.mode_run
        self.mode_cls   = args.mode_cls
        self.idx_trn    = args.idx_trn
        self.idx_val    = args.idx_val
        self.epochs     = args.epochs
        self.crop3d     = [int(xx) for xx in args.crop3d.split('x')]
        if self.idx_trn is not None:
            self.wdir       = os.path.dirname(self.idx_trn)
        else:
            self.wdir = None
        self.is_zerobg  = args.is_zerobg
        self.iters_per_epoch = args.iters_per_epoch
        self.iters_val  = args.iters_val
        self.num_wrks   = args.num_wrks
        self.infer_shape= tuple([int(xx) for xx in args.infer_shape.split('x')])
        self.infer_crop = tuple([int(xx) for xx in args.infer_crop.split('x')])
        self.infer_pad  = args.infer_pad
        self.infer_crop_pad = tuple([xx + 2*self.infer_pad for xx in self.infer_crop])
        #
        self.num_cls      = 7
        self.rnd_siz_scl  = (0.8, 1.2)
        self.rnd_val_scl  = (0.9, 1.1)
        self.rnd_val_sht  = (-0.08, +0.08)
        #
        self.batch_size     = args.batch_size
        self.smpl_per_img   = args.smpl_per_img
        self.inp_shape      = (self.crop3d[0], self.crop3d[1], self.crop3d[2], 1)
        #
        self.class_weights = [1, 1, 2, 2, 2, 2, 2]
        self.path_model = '{}_fcn3d_valloss_{}_bg{}.h5' \
            .format(self.idx_trn, args.crop3d, self.is_zerobg)
        self.path_log = '{}-log.csv'.format(self.path_model)
        self.path_log_dir = os.path.join(os.path.dirname(self.path_model), 'logs',
                                         os.path.splitext(os.path.basename(self.path_model))[0])
        #
        if args.model is not None:
            self.path_model = args.model
        self.path_img       = args.img
        self.path_msk_lung  = args.lung
        self.infer_out      = args.infer_out
        self.infer_idx      = args.infer_idx
        #
        self.dir_val_cache = '{}-val-cache-b{}'.format(self.path_model, self.batch_size)
        os.makedirs(self.dir_val_cache, exist_ok=True)
        self.path_config = '{}-cfg.json'.format(self.path_model)
        logging.info(' :: config:\n\t{}'.format(self.to_json()))
        self.paths_trn  = self._load_idx(self.idx_trn)
        self.paths_val  = self._load_idx(self.idx_val)
        with open(self.path_config, 'w') as f:
            f.write(self.to_json())
    def _load_idx(self, path_idx):
        if path_idx is None:
            return None
        wdir = os.path.dirname(path_idx)
        data = pd.read_csv(path_idx)
        ret = []
        for ii, irow in data.iterrows():
            ret.append({
                'img':  os.path.join(wdir, irow['path_img']),
                'cls':  os.path.join(wdir, irow['path_lesion']),
                'msk': os.path.join(wdir, irow['path_lung']),
            })
        return ret
    def to_json(self):
        return json.dumps(self.__dict__, indent=4)

def get_or_load_model(cfg) -> keras.models.Model:
    """
    :type cfg: ConfigTrain
    """
    model = build_model(cfg)
    model.summary()
    path_model_restart = cfg.path_model
    if not os.path.isfile(path_model_restart):
        logging.warning(' [!!!!] no model found on path: [{}]'.format(path_model_restart))
    else:
        pref = time.strftime('%Y.%m.%d-%H.%M.%S')
        path_model_bk = '%s-%s.bk' % (cfg.path_model, pref)
        if cfg.mode_run == 'train':
            logging.info(' @backup previous saved model file: [{}]'.format(path_model_bk))
            shutil.copy(cfg.path_model, path_model_bk)
        logging.info(' [***] found model trained weights from file [{}]'.format(path_model_restart))
        model.load_weights(path_model_restart, by_name=True)
    optim = keras.optimizers.Adam(lr=0.0002)
    if cfg.mode_cls == 'segm':
        model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['binary_accuracy'])
    else:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def _task_prepare_batch(pdata):
    idx, num_batches, path_x, path_y, cfg = pdata
    cfg = copy.copy(cfg)
    ptr_generator = data_generator(cfg, is_train=False, is_randomize=False)
    data_x, data_y = next(ptr_generator)
    logging.info(' [*] [{}/{}] generate validation batch: [{}]'.format(idx, num_batches, path_x))
    np.save(path_x, data_x)
    np.save(path_y, data_y)
    return True

class DataGeneratorCache(keras.utils.Sequence):
    def __init__(self, dir_cache, batch_size):
        self.dir_cache  = dir_cache
        self.batch_size = batch_size
        if not os.path.isdir(dir_cache):
            raise FileNotFoundError('Cant find data-cache directory: [{}]'.format(self.dir_cache))
        # self.path_batches = glob.glob('{}/batch_n*_b{}_x.npy'.format(self.dir_cache, self.batch_size))
        self.path_batches = glob.glob('{}/batch_n*_x.npy'.format(self.dir_cache, self.batch_size))
        self.num_data = len(self.path_batches)
        if self.num_data < 2:
            raise Exception('too small validation batches (batch={}) in directory [{}]'.format(self.batch_size, self.path_batches))
    def __len__(self):
        return self.num_data
    def __getitem__(self, idx):
        path_x = self.path_batches[idx]
        path_y = path_x.replace('_x.npy', '_y.npy')
        data_x = np.load(path_x)
        data_y = np.load(path_y)
        return data_x, data_y

def prepare_generator_cached(cfg, is_threaded=True) -> keras.utils.Sequence:
    """
    :type cfg: Config
    """
    paths_datax = glob.glob('{}/batch_n*_b{}_x.npy'.format(cfg.dir_val_cache, cfg.batch_size))
    paths_datay = glob.glob('{}/batch_n*_b{}_y.npy'.format(cfg.dir_val_cache, cfg.batch_size))
    num_x = len(paths_datax)
    num_y = len(paths_datay)
    if num_x != num_y:
        raise Exception('Cache inconsistent: #data-x({}) != #data-y({}), cache = [{}]'.format(num_x, num_y, cfg.dir_val_cache))
    if num_x < 2:
        logging.warning('\t:: to small #data ({}): clean & rebuild:'.format(num_x))
        shutil.rmtree(cfg.dir_val_cache, ignore_errors=True)
        os.makedirs(cfg.dir_val_cache)
        #
        num_batches = cfg.iters_val
        t1 = time.time()
        lst_task_data = []
        # for ii, (data_x, data_y) in enumerate(data_generator_val):
        if not is_threaded:
            data_generator_val = data_generator(cfg, is_train=True, is_randomize=False)
        else:
            data_generator_val = False
        for ii in range(num_batches):
            path_x = os.path.join(cfg.dir_val_cache, 'batch_n{:06d}_b{}_x.npy'.format(ii, cfg.batch_size))
            path_y = os.path.join(cfg.dir_val_cache, 'batch_n{:06d}_b{}_y.npy'.format(ii, cfg.batch_size))
            lst_task_data.append([ii, num_batches, path_x, path_y, copy.copy(cfg)])
            if not is_threaded:
                data_x, data_y = next(data_generator_val)
                logging.info(' [*] [{}/{}] generate validation batch: [{}]'.format(ii, num_batches, path_x))
                np.save(path_x, data_x)
                np.save(path_y, data_y)
            # if ii >= num_batches:
            #     break
        dt = time.time() - t1
        if is_threaded:
            logging.info(' [*] start threaded data generation: #tasks/#threads = {}/{}'.format(len(lst_task_data), cfg.num_wrks))
            pool = mp.ProcessPoolExecutor(max_workers=cfg.num_wrks)
            lst_ret = pool.map(_task_prepare_batch, lst_task_data)
            pool.shutdown(wait=True)
        logging.info('\t\t... [done], time-val-generation = {:0.2f} (s)'.format(dt))
    else:
        logging.info(' [**] Found validation cache, loading from ... [{}]'.format(cfg.dir_val_cache))
    data_cache = DataGeneratorCache(cfg.dir_val_cache, cfg.batch_size)
    return data_cache

##################################################
def inference_probmap(cfg, model, img3d, shape_crop=(128, 128, 128), pad=32) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    :type cfg: Config
    :type img3d: np.ndarray
    :type shape_crop: tuple
    """
    # shape_crop = cfg.infer_crop
    # pad = cfg.infer_pad
    shape_crop_bad = cfg.infer_crop_pad #(np.array(shape_crop) + 2 * pad).tolist()
    img3d_pad = np.pad(img3d, [[pad, pad], [pad, pad], [pad, pad]], mode='constant')
    img_shape_pad = img3d_pad.shape
    #
    lst_xyz = [list(range(pad, xx - pad, yy)) for xx, yy in zip(img_shape_pad, shape_crop)]
    pmap_pad = np.zeros(list(img_shape_pad) + [cfg.num_cls], dtype=np.float32)
    if isinstance(model, str):
        model = build_model(cfg, inp_shape=list(shape_crop) + [1])
        model.load_weights(model, by_name=True)
        model.summary()
    for xxi, xx in enumerate(lst_xyz[0]):
        for yyi, yy in enumerate(lst_xyz[1]):
            for zzi, zz in enumerate(lst_xyz[2]):
                progress_idx = [(x1, len(x2)) for x1, x2 in zip([xxi, yyi, zzi], lst_xyz)]
                logging.info('\t{} * processing'.format(progress_idx))
                imcrop = img3d_pad[xx-pad:xx-pad+shape_crop_bad[0],
                         yy-pad:yy-pad+shape_crop_bad[1],
                         zz-pad:zz-pad+shape_crop_bad[2]]
                ret = model.predict_on_batch(imcrop[None, :, :, :, None])
                pmap_pad[xx-pad:xx-pad+shape_crop_bad[0],
                         yy-pad:yy-pad+shape_crop_bad[1],
                         zz-pad:zz-pad+shape_crop_bad[2]] = ret[0]
    pmap_cls = np.argsort(-pmap_pad, axis=-1)[:, :, :, 0]
    pmap_val = -np.sort(-pmap_pad, axis=-1)[:, :, :, 0]
    pmap_cls = pmap_cls[pad:-pad, pad:-pad, pad:-pad].astype(np.uint8)
    pmap_val = (255.*pmap_val[pad:-pad, pad:-pad, pad:-pad]).astype(np.uint8)
    return pmap_cls, pmap_val


def _clone_nifti(im3d, nifti_inp) -> nib.Nifti1Image:
    """
    :type im3d: np.ndarray
    :type nifti_inp: nib.Nifti1Image
    """
    if isinstance(nifti_inp, str):
        nifti_inp = nib.load(nifti_inp)
    assert (im3d.shape == nifti_inp.shape)
    meta_affine = nifti_inp.affine
    meta_header = nifti_inp.header
    meta_header.set_data_dtype(im3d.dtype)
    ret_nifti = nib.Nifti1Image(im3d, meta_affine, header=meta_header)
    return ret_nifti


def run_inference(cfg, model, path_out_nii = None):
    """
    :type cfg: Config
    :type model: keras.models.Model
    """
    logging.info(' [*] processing input image [{}]'.format(cfg.path_img))
    path_img  = cfg.path_img
    path_lung = cfg.path_msk_lung
    img3d_inp = _norm_img3d(nib.load(path_img).get_data().astype(np.float32))
    msk_lung  = nib.load(path_lung).get_data()
    shape0 = img3d_inp.shape
    if cfg.infer_shape != shape0:
        logging.info('\tresize input image {} -> {}'.format(img3d_inp.shape, cfg.infer_shape))
        img3d_inp = sk.transform.resize(img3d_inp, cfg.infer_shape, order=2)
    if shape0 != msk_lung.shape:
        logging.info('\tresize lungs image {} -> {}'.format(msk_lung.shape, shape0))
        msk_lung = sk.transform.resize(msk_lung, shape0, order=0)
    map_cls, map_val = inference_probmap(cfg, model, img3d_inp, shape_crop=cfg.infer_crop, pad=cfg.infer_pad)
    if map_cls.shape != shape0:
        map_cls = sk.transform.resize(map_cls, shape0, order=0, preserve_range=True).astype(np.uint8)
        map_val = sk.transform.resize(map_val, shape0, order=1, preserve_range=True).astype(np.uint8)
    map_cls[msk_lung < 0.5] = 0
    map_val[msk_lung < 0.5] = 0
    nii_cls = _clone_nifti(map_cls, path_img)
    nii_val = _clone_nifti(map_val, path_img)
    if path_out_nii is not None:
        path_out_val = path_out_nii + '-prob.nii.gz'
        logging.info('\t:: export cls/prob-map data: {}/{}'.format(path_out_nii, path_out_val))
        nib.save(nii_cls, path_out_nii)
        nib.save(nii_val, path_out_val)
    return nii_cls, nii_val


##################################################
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode_run',   type=str, required=False, default=Config.modes_run[0],
                        help='running modes, available modes: {}'.format(Config.modes_run))
    parser.add_argument('--mode_cls',   type=str, required=False, default='trn',
                        help='classification modes, available modes: {}'.format(Config.modes_cls))
    parser.add_argument('--idx_trn',    type=str, required=False,  default=None, help='path to train csv')
    parser.add_argument('--idx_val',    type=str, required=False, default=None, help='path to validation csv')
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='batch size')
    parser.add_argument('--smpl_per_img', type=int, required=False, default=4, help='batch size')
    parser.add_argument('--epochs',     type=int, required=False, help='#epochs', default=100)
    parser.add_argument('--iters_per_epoch', type=int, required=False, help='#epochs', default=200)
    parser.add_argument('--iters_val',  type=int, required=False, help='#epochs', default=500)
    parser.add_argument('--is_debug',   action='store_true', help='debug mode')
    parser.add_argument('--is_zerobg',  action='store_true', help='zero loss in non-lung voxels')
    parser.add_argument('--crop3d',     type=str, required=False, default='96x96x64',
                        help='input image shape in str-format WidthxHeightxChannels, like 96x96x64')
    parser.add_argument('--num_wrks',   type=int, required=False, help='#workers', default=3)
    parser.add_argument('--out_prefix', type=str, required=False, default=None, help='additional prefix for output inferenced lesion-mask')
    #
    parser.add_argument('--infer_shape', type=str, required=False, help='#workers', default='512x512x256')
    parser.add_argument('--infer_crop',  type=str, required=False, help='#workers', default='128x128x64')
    parser.add_argument('--infer_pad',   type=int, required=False, help='#workers', default=32)
    parser.add_argument('--model',       type=str, required=False, help='Model path for inference', default=None)
    parser.add_argument('--img',         type=str, required=False, help='Input image for processing', default=None)
    parser.add_argument('--lung',        type=str, required=False, help='Input lungs mask for processing', default=None)
    parser.add_argument('--infer_out',   type=str, required=False, help='Output path for class-mask', default=None)
    parser.add_argument('--infer_idx',   type=str, required=False, help='Inference index file', default=None)
    args = parser.parse_args()
    # (1) prepare config
    cfg = Config(args)
    # (2) prepare train/validation data-generators
    if cfg.mode_run in ['train', 'val']:
        ptr_generator_trn = data_generator(cfg, is_train=True, is_randomize=True)
        # ptr_generator_val = data_generator(cfg, is_train=True, is_randomize=False)
        ptr_generator_val_cached = prepare_generator_cached(cfg, is_threaded=True)
        model = get_or_load_model(cfg)
    else:
        ptr_generator_trn = None
        ptr_generator_val = None
        ptr_generator_val_cached = None
        model = None
    # (3) run stage: train/validataion/test/ ... / inference
    logging.info('\t[MODE]-----({})---'.format(cfg.mode_run))
    if cfg.mode_run == 'train':
        model.fit_generator(
            generator=ptr_generator_trn,
            steps_per_epoch=cfg.iters_per_epoch,
            epochs=cfg.epochs,
            validation_data=ptr_generator_val_cached,
            use_multiprocessing=(cfg.num_wrks > 1),
            workers=cfg.num_wrks,
            # max_queue_size=cfg.num_wrks + 2,  # FIXME: +2, why ???
            callbacks=[
                keras.callbacks.ModelCheckpoint(cfg.path_model, verbose=True, save_best_only=True, monitor='val_loss'),
                keras.callbacks.TensorBoard(cfg.path_log_dir, histogram_freq=False, write_graph=True, write_images=True),
                keras.callbacks.CSVLogger(cfg.path_log, append=True)
            ])
        print('-')
    elif cfg.mode_run == 'val':
        for ii, (xx, yy) in enumerate(ptr_generator_val_cached):

            print('-')
    elif cfg.mode_run == 'infer':
        t1 = time.time()
        model = build_model(cfg, inp_shape=list(cfg.infer_crop_pad) + [1])
        model.summary()
        logging.info('\tloading model weights from [{}]'.format(cfg.path_model))
        model.load_weights(cfg.path_model, by_name=True)
        ret_cls, ret_pmap = run_inference(cfg, model, path_out_nii=cfg.infer_out)
        dt = time.time() - t1
        logging.info('\t\tdone... dt ~ {:0.3f} (s) * [{}]'.format(dt, cfg.path_img))
        # print('-')
    elif cfg.mode_run == 'infer2':

        model = build_model(cfg, inp_shape=list(cfg.infer_crop_pad) + [1])
        model.summary()
        logging.info('\tloading model weights from [{}]'.format(cfg.path_model))
        model.load_weights(cfg.path_model, by_name=True)
        dir_data = os.path.dirname(cfg.infer_idx)
        data_idx = pd.read_csv(cfg.infer_idx)
        num_idx = len(data_idx)
        path_model = cfg.path_model
        for ii, idata in data_idx.iterrows():
            path_img = os.path.join(dir_data, idata['path_img'])
            path_lung = os.path.join(dir_data, idata['path_lung'])
            if 'path_lesion' in idata.keys():
                path_lesion = os.path.join(dir_data, idata['path_lesion'])
                model_prefix = str(os.path.splitext(os.path.basename(cfg.path_model))[0].split('-trn-')[1])
                cfg.infer_out = path_lesion + '-predict-' + model_prefix + '.nii.gz'
            else:
                name = os.path.splitext(os.path.join(idata['path_lung']))[0]
                if args.out_prefix is not None:
                    name = name.replace('-lungs', '-lesion5-{}'.format(args.out_prefix))
                else:
                    name = name.replace('-lungs', '-lesion5')
                path_lesion = os.path.join(dir_data, name + '.nii.gz')
                cfg.infer_out = path_lesion
            logging.info('[{}/{}] -> [{}]'.format(ii, num_idx, path_img))
            cfg.path_img = path_img
            cfg.path_msk_lung = path_lung
            t1 = time.time()
            if not os.path.isfile(cfg.infer_out):
                is_ok = True
                if not os.path.isfile(cfg.path_img):
                    logging.error('!!!!! cant find image file, skip... [{}]'.format(cfg.path_img))
                    is_ok = False
                if not os.path.isfile(cfg.path_msk_lung):
                    logging.error('!!!!! cant find lung image file, skip... [{}]'.format(cfg.path_msk_lung))
                    is_ok = False
                if is_ok:
                    ret_cls, ret_pmap = run_inference(cfg, model, path_out_nii=cfg.infer_out)
            else:
                logging.warning('\t!!! predicted file exist, skip... [{}]'.format(cfg.infer_out))
            dt = time.time() - t1
            logging.info('\t\tdone... dt ~ {:0.3f} (s) * [{}]'.format(dt, cfg.path_img))
            print('-')
        # print('-')
    else:
        raise NotImplementedError()

