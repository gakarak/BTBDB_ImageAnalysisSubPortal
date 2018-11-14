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
import skimage.morphology
import scipy.ndimage.morphology as smorph
import concurrent.futures as mp
import skimage.transform


def resize_nii(pathNii, newSize=(33, 33, 33), parOrder=4, parMode='edge', parPreserveRange=True):
    if isinstance(pathNii,str):# or isinstance(pathNii,unicode):
        tnii = nib.load(pathNii)
    else:
        tnii = pathNii
    timg = tnii.get_data()
    oldSize = timg.shape
    dataNew = skimage.transform.resize(timg, newSize, order=parOrder, preserve_range=parPreserveRange, mode=parMode)
    affineOld = tnii.affine.copy()
    affineNew = tnii.affine.copy()
    k20_Old = float(oldSize[2]) / float(oldSize[0])
    k20_New = float(newSize[2]) / float(newSize[0])
    for ii in range(3):
        tCoeff = float(newSize[ii]) / float(oldSize[ii])
        if ii == 2:
            tCoeff = (affineNew[0, 0] / affineOld[0, 0]) * (k20_Old / k20_New)
        affineNew[ii, ii] *= tCoeff
        affineNew[ii,  3] *= tCoeff
    retNii = nib.Nifti1Image(dataNew, affineNew, header=tnii.header)
    return retNii

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


def task_process_lung_mask(pdata):
    idx, num_idx, [path_lung_mask, path_out, size] = pdata
    if os.path.isfile(path_out):
        logging.warning('\t[!!!!] dilated lung exist, skip ... [{}]'.format(path_out))
        return True
    logging.info('\t[{}/{}] dilating 3d-mask ({}): [{}]'.format(idx, num_idx, size, path_out))
    t1 = time.time()
    msk3d = (nib.load(path_lung_mask).get_data()>0)
    elem3d = skimage.morphology.ball(size, dtype=np.bool)
    msk3d_dilated = smorph.binary_dilation(msk3d, elem3d).astype(np.float32)
    nii_out = _clone_nifti(msk3d_dilated, path_lung_mask)
    nib.save(nii_out, path_out)
    dt = time.time() - t1
    logging.info('\t\t[{}/{}] done... dt ~ {:0.2f} (s)'.format(idx, num_idx, dt))
    return True


def _ext_resize_lung_mask(path_nii_inp, inp_shape, new_shape):
    pass



def task_process_lung_mask_and_resize(pdata):
    idx, num_idx, [path_lung_mask, path_out, morph_size] = pdata
    lst_ext_sizes = [192, 320]
    if os.path.isfile(path_out):
        logging.warning('\t[!!!!] dilated lung exist, skip ... [{}]'.format(path_out))
        return True
    logging.info('\t[{}/{}] dilating 3d-mask ({}): [{}]'.format(idx, num_idx, morph_size, path_out))
    t1 = time.time()
    msk3d = (nib.load(path_lung_mask).get_data()>0)
    elem3d = skimage.morphology.ball(morph_size, dtype=np.bool)
    msk3d_dilated = smorph.binary_dilation(msk3d, elem3d).astype(np.float32)
    nii_out = _clone_nifti(msk3d_dilated, path_lung_mask)
    nib.save(nii_out, path_out)
    dt = time.time() - t1
    logging.info('\t\t[{}/{}] done... dt ~ {:0.2f} (s)'.format(idx, num_idx, dt))
    return True


def get_out_path(pinp, psiz):
    path_basic = pinp[:-7] # remove .nii.gz
    path_ret = '{}_dr{}.nii.gz'.format(path_basic, psiz)
    return path_ret


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    num_threads = 1
    m_radius = 11
    fidx = '/media/data10T_1/datasets/CRDF_3/idx-good-all-512x512x256.txt'
    wdir = os.path.dirname(fidx)
    data_csv = pd.read_csv(fidx)
    #
    paths_lungs = [os.path.join(wdir, xx) for xx in data_csv['path_lung']]
    num_lungs = len(paths_lungs)
    lst_task_data = [(xxi, num_lungs, [xx, get_out_path(xx, m_radius), m_radius]) for xxi, xx in enumerate(paths_lungs)]
    logging.info(' start threaded processing: #tasks/#threads = {}/{}'.format(num_lungs, num_threads))
    t1 = time.time()
    if (num_lungs < 2) or (num_threads < 2):
        lst_res = [task_process_lung_mask(xx) for xx in lst_task_data]
    else:
        pool = mp.ProcessPoolExecutor(max_workers=num_threads)
        lst_res = list(pool.map(task_process_lung_mask, lst_task_data))
        pool.shutdown(wait=True)
    dt = time.time() - t1
    logging.info('\t... done, dt = {:0.2f} (s)'.format(dt))

    print('-')