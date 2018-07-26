#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar (Alexander Kalinovsky)'

import os

import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import time

from app.core.segmct import segmentLungs25D
import pandas as pd
from app.core.preprocessing import resizeNii, resize3D
# from run00_common import BatcherOnImageCT3D, split_list_by_blocks


#########################################
if __name__ == '__main__':
    path_model_lung = '/home/ar/github.com/BTBDB_ImageAnalysisSubPortal.git/experimental_data/models/fcnn_ct_lung_segm_2.5d_tf'
    # path_model_lesion = '../../../experimental_data/models/fcnn_ct_lesion_segm_3d_tf/'
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_idx', type=str, default=None, required=True, help='path to index with images')
    args = parser.parse_args()
    #
    path_idx = args.path_idx
    shape_4lung = (256, 256, 64)
    wdir = os.path.dirname(path_idx)
    data_csv = pd.read_csv(path_idx)
    paths_img = [os.path.join(wdir, xx) for xx in data_csv['path_img']]
    num_img = len(paths_img)
    #
    for ii, path_nii in enumerate(paths_img):
        print(' [{}/{}] * {}'.format(ii, num_img, paths_img))
        t1 = time.time()
        dataNii = nib.load(path_nii)
        shape_orig = dataNii.shape
        nii_resiz_4lung = resizeNii(dataNii, shape_4lung)
        lungMask = segmentLungs25D(nii_resiz_4lung,
                                   dirWithModel=path_model_lung,
                                   pathOutNii=None,
                                   outSize=shape_orig,
                                   # outSize=shape4Lung,
                                   threshold=0.5)
        dt = time.time() - t1
        path_out_lungs = '{}-lungs.nii.gz'.format(path_nii)
        print('\t\tdone, dt={:0.2f} (s), export result to [{}]'.format(dt, path_out_lungs))
        nib.save(lungMask, path_out_lungs)
