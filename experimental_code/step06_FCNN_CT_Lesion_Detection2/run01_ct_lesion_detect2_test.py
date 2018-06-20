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

from app.core.segmct.fcnn_lesion3dv2 import buildModelFCN3D, Inferencer, lesion_id2rgb, get_overlay_msk


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
    msk_on_img = get_overlay_msk(timg2d, tmsk2d)

    plt.subplot(1, 3, 1)
    plt.imshow(timg2d)
    plt.subplot(1, 3, 2)
    plt.imshow(tmsk2d)
    plt.subplot(1, 3, 3)
    plt.imshow(msk_on_img)
    plt.show()
    print('-')