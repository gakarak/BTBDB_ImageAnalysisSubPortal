#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar (Alexander Kalinovsky)'

import os
import glob
import time
import shutil
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
import tensorflow as tf
import keras.backend as K
from collections import OrderedDict

#############################################
lesion_id2name = OrderedDict({
    0: 'background',
    1: 'class_1',
    2: 'class_2',
    3: 'class_3',
    4: 'class_4',
    5: 'class_5',
    6: 'class_6'
})
lesion_name2id = OrderedDict({vv:kk for kk, vv in lesion_id2name.items()})

#############################################
lesion_id2rgb = {
    0: [0, 0, 0],
    1: [1, 0, 0],
    2: [0, 1, 0],
    3: [0, 0, 1],
    4: [1, 1, 0],
    5: [0, 1, 1],
    6: [1, 0, 1],
    7: [0.7, 0.7, 0.7],
}

def get_overlay_msk(img2d, msk_lbl, alpha = 0.5, dct_colors=lesion_id2rgb):
    img2d = img2d.astype(np.float32)
    if img2d.max() > 1:
        img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min())
    if img2d.ndim < 3:
        img2d = np.tile(img2d[..., np.newaxis], 3)
    msk_bin = (msk_lbl > 0)
    msk_rgb = np.tile(msk_bin[..., np.newaxis], 3)
    #
    img_bg = (msk_rgb == False) * img2d
    ret = img_bg.copy()
    for kk, vv in dct_colors.items():
        if kk < 1:
            continue
        tmp_msk = (msk_lbl == kk)
        tmp_msk_rgb = np.tile(tmp_msk[..., np.newaxis], 3)
        tmp_img_orerlay = alpha * np.array(vv) * tmp_msk_rgb
        tmp_img_original = (1 - alpha) * tmp_msk_rgb * img2d
        ret += tmp_img_orerlay + tmp_img_original
    return ret

#############################################
def split_list_by_blocks(lst, psiz):
    tret = [lst[x:x + psiz] for x in range(0, len(lst), psiz)]
    return tret


def buildModelFCN3D(inpShape=(256, 256, 128, 1), numCls=2, numConv=1, kernelSize=3, numFlt=16, ppad='same', numSubsampling=5,
                  isUnet=True, isDebug=False, isSkipFirstConnection=False):
    dataInput = kl.Input(shape=inpShape)
    fsiz = (kernelSize, kernelSize, kernelSize)
    psiz = (2, 2, 2)
    x = dataInput
    # -------- Encoder --------
    lstMaxPools = []
    for cc in range(numSubsampling):
        for ii in range(numConv):
            x = kl.Conv3D(filters=numFlt * (2**cc), kernel_size=fsiz, padding=ppad, activation='relu')(x)
        lstMaxPools.append(x)
        x = kl.MaxPooling3D(pool_size=psiz)(x)
    # -------- Decoder --------
    for cc in range(numSubsampling):
        for ii in range(numConv):
            x = kl.Conv3D(filters=numFlt * (2 ** (numSubsampling - 1 -cc)), kernel_size=fsiz, padding=ppad, activation='relu')(x)
        x = kl.UpSampling3D(size=psiz)(x)
        if isUnet:
            if isSkipFirstConnection and (cc == (numSubsampling - 1)):
                continue
            x = kl.concatenate([x, lstMaxPools[-1 - cc]], axis=-1)
    # final convs
    for cc in range(numSubsampling):
        x = kl.Conv3D(filters=numFlt, kernel_size=fsiz, padding=ppad, activation='relu')(x)
    # 1x1 Convolution: emulation of Dense layer
    if numCls == 2:
        x = kl.Conv3D(filters=1, kernel_size=(1, 1, 1), padding='valid', activation='sigmoid', name='conv3d_out_c{}'.format(numCls))(x)
        x = kl.Reshape([inpShape[0], inpShape[1], inpShape[2]])(x)
    else:
        x = kl.Conv3D(filters=numCls, kernel_size=(1, 1, 1), padding='valid', activation='linear', name='conv3d_out_c{}'.format(numCls))(x)
        x = kl.Reshape([-1, numCls])(x)
        x = kl.Activation('softmax')(x)
        x = kl.Reshape([inpShape[0], inpShape[1], inpShape[2], numCls])(x)
    retModel = keras.models.Model(dataInput, x)
    if isDebug:
        retModel.summary()
        fimg_model = 'model_graph_FCNN3D.png'
        keras.utils.plot_model(retModel, fimg_model, show_shapes=True)
        plt.imshow(plt.imread(fimg_model))
        plt.show()
    return retModel

######################################################
class Inferencer:

    def __init__(self, inp_shape3d=(256, 256, 64, 1), num_cls = 7):
        self.inp_shape3d = inp_shape3d
        self.msk_shp = self.inp_shape3d[:3]
        self.num_cls = num_cls
        self.model = None

    def load_model(self, path_model, is_add_to_class = True):
        if os.path.isdir(path_model):
            lst_models = sorted(glob.glob('{}/*.h5'.format(path_model)))
            if len(lst_models) < 1:
                raise Exception('Cant find model files (*.h5) in dorectory [{}]'.format(path_model))
            path_model = lst_models[0]
        if not os.path.isfile(path_model):
            raise Exception('Cant find Model-file [%s]' % path_model)
        model = buildModelFCN3D(self.inp_shape3d, self.num_cls)
        model.load_weights(path_model, by_name=True)
        if is_add_to_class:
            self.model = model
        return model

    def get_img3d(self, pimg3d, ptype=np.float32, porder=1):
        if isinstance(pimg3d, str):
            pimg3d = nib.load(pimg3d).get_data().astype(ptype)
        elif isinstance(pimg3d, nib.nifti1.Nifti1Image):
            pimg3d = pimg3d.get_data().astype(ptype)
        if self.inp_shape3d is not None:
            pshape3d = tuple(self.inp_shape3d)[:3]
            if pimg3d.shape != pshape3d:
                p1 = pimg3d.min()
                p2 = pimg3d.max()
                pimg3d = (pimg3d - p1) / (p2 - p1 + 0.0001)
                pimg3d = sk.transform.resize(pimg3d, pshape3d, order=porder)
                pimg3d = p1 + (p2 - p1) * pimg3d
        # FIXME: this is KOSTIL for Vitaly prepared data!!!
        if pimg3d.min() < -4000.:
            pimg3d += 1024.
            # if tmp is not None:
            #     print('!!!! FFUCK: {}'.format(tmp))
        return pimg3d

    def norm_img3d(self, pimg3d, p1=-2400, p2=1200):
        pimg3d = (pimg3d - p1) / (p2 - p1)
        pimg3d[pimg3d < 0.0] = 0.0
        pimg3d[pimg3d > 1.0] = 1.0
        pimg3d = (2. * pimg3d) - 1
        return pimg3d

    def inference(self, lst_data, batchSize=2):
        if self.model is None:
            raise Exception('Model is not loaded... load model before call inference()')
        if len(lst_data)>0:
            lst_of_img = []
            # (1) load into memory
            if isinstance(lst_data[0], str) or isinstance(lst_data[0], nib.nifti1.Nifti1Image):
                for ii in lst_data:
                    timg3d = self.get_img3d(ii)
                    timg3d = np.expand_dims(self.norm_img3d(timg3d), axis=-1)
                    lst_of_img.append(timg3d)
            else:
                lst_of_img = lst_data
            # (2) check shapes
            tsetShapes = set()
            for ii in lst_of_img:
                tsetShapes.add(ii.shape)
            if len(tsetShapes)>1:
                raise Exception('Shapes of images must be equal sized')
            if self.inp_shape3d not in tsetShapes:
                raise Exception('Model input shape and shapes of input images is not equal!')
            # (3) inference
            num_img = len(lst_of_img)
            lst_idx = range(num_img)
            split_idx = split_list_by_blocks(lst_idx, batchSize)
            ret = []
            for ss in split_idx:
                data_x = np.zeros([len(ss)] + list(self.inp_shape3d), dtype=np.float32)
                for ii, ssi in enumerate(ss):
                    data_x[ii] = lst_of_img[ssi]
                ret_prob = self.model.predict_on_batch(data_x)
                for ii in range(ret_prob.shape[0]):
                    tlbl = np.argmax(ret_prob[ii], axis=-1)
                    ret.append(tlbl.reshape(self.msk_shp))
            return ret
        else:
            return []

#########################################
if __name__ == '__main__':
    pass