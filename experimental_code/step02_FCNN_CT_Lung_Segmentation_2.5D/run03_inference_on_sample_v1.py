#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar (Alexander Kalinovsky)'

import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from run00_common import BatcherOnImageCT3D, split_list_by_blocks

#########################################
if __name__ == '__main__':
    pathDirWithModels = '../../experimental_data/models/fcnn_ct_lung_segm_2.5d'
    batcherInfer = BatcherOnImageCT3D()
    batcherInfer.loadModelForInference(pathModelJson=pathDirWithModels, pathMeanData=pathDirWithModels)
    batcherInfer.model.summary()
    print (batcherInfer)
    lstPathNifti=[
        '../../experimental_data/resize-256x256x64/0009-256x256x64.nii.gz',
        '../../experimental_data/resize-256x256x64/0026-256x256x64.nii.gz',
        '../../experimental_data/resize-256x256x64/0177-256x256x64.nii.gz',
    ]
    ret = batcherInfer.inference(lstPathNifti, batchSize=2)
    numRet = len(ret)
    plt.figure()
    for ii,msk in enumerate(ret):
        plt.subplot(2, numRet, ii + 1)
        plt.imshow(msk[:,:,msk.shape[-1]/2])
        timg = nib.load(lstPathNifti[ii]).get_data()
        plt.subplot(2, numRet, numRet +ii + 1)
        plt.imshow(timg[:, :, timg.shape[-1] / 2])
    plt.show()
    print ('---')