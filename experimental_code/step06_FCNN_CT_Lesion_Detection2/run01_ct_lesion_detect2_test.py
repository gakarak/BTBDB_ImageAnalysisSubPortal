#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
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

dct_colors = {
    0: [0, 0, 0],
    1: [1, 0, 0],
    2: [0, 1, 0],
    3: [0, 0, 1],
    4: [1, 1, 0],
    5: [0, 1, 1],
    6: [1, 0, 1],
    7: [0.7, 0.7, 0.7],
}

from app.core.segmct.fcnn_lesion3dv2 import buildModelFCN3D, Inferencer

def _get_overlay_msk(img2d, msk_lbl, alpha = 0.5, dct_colors=dct_colors):
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

if __name__ == '__main__':
    path_model = '../../experimental_data/models/fcnn_ct_lesion_segm_3dv2_tf/idx-lesions-trn-256x256x64.txt_model_fcn3d_lesions.h5'
    path_ct3d = '../../experimental_data/dataentry_test0/case-2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0/study-dd10657e-f2c3-48ba-87d6-b5f3fc40c752/series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2-CT.nii.gz'
    #
    infrn = Inferencer()
    infrn.load_model(path_model)
    tmsk = infrn.inference([path_ct3d])[0]
    timg = infrn.get_img3d(path_ct3d)

    timg2d = timg[:, :, 32]
    tmsk2d = tmsk[:, :, 32]
    msk_on_img = _get_overlay_msk(timg2d, tmsk2d)

    plt.subplot(1, 3, 1)
    plt.imshow(timg2d)
    plt.subplot(1, 3, 2)
    plt.imshow(tmsk2d)
    plt.subplot(1, 3, 3)
    plt.imshow(msk_on_img)
    plt.show()
    print('-')