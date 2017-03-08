#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import time
import numpy as np
import json

import nibabel as nib

import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skolor
import pandas as pd
import matplotlib.pyplot as plt

try:
   import cPickle as pickle
except:
   import pickle

import keras
from keras.utils.visualize_util import plot as kplot

from run00_common import BatcherOnImageCT3D, split_list_by_blocks

######################################################
if __name__=='__main__':
    print (':: local directory: %s' % os.getcwd())
    #
    # fidxTrain = '/mnt/data1T/datasets/CRDF/TB_5_Classes/TB_sub_1_5-resize-128x128x64/idx.txt-train.txt'
    # fidxVal   = '/home/ar/datasets/CRDF/CT_with_segm_mask_v1/resize-128x128x64/idx.txt-val.txt'
    fidxTrain = '/home/ar/datasets/CRDF/CT_with_segm_mask_v1/resize-128x128x64/idx.txt-train.txt'
    fidxVal = '/mnt/data1T/datasets/CRDF/TB_5_Classes/TB_sub_1_5-resize-128x128x64/idx.txt-val.txt'
    parIsTheanoShape = True
    parIsTheanoShape = True
    parBatchSizeVal  = 16
    batcherTrain = BatcherOnImageCT3D(pathDataIdx=fidxTrain,
                                      isTheanoShape=parIsTheanoShape)
    batcherVal = BatcherOnImageCT3D(pathDataIdx=fidxVal,
                                    pathMeanData=batcherTrain.pathMeanData,
                                    isTheanoShape=parIsTheanoShape)
    print (':: Train data: %s' % batcherTrain)
    print (':: Val   data: %s' % batcherVal)
    #
    wdir = os.path.dirname(fidxTrain)
    # modelTrained = BatcherOnImageCT3D.loadModelFromDir(wdir, paramFilter='950')
    # modelTrained = BatcherOnImageCT3D.loadModelFromDir(wdir)
    # modelTrained = BatcherOnImageCT3D.loadModelFromDir(wdir, paramFilter='adagrad')
    modelTrained = BatcherOnImageCT3D().loadModelFromDir(wdir, paramFilter='adam')
    modelTrained.summary()
    #
    t0 = time.time()
    lstIdxSplit = split_list_by_blocks(range(batcherVal.numImg), parBatchSizeVal)
    numSplit = len(lstIdxSplit)
    for ii,ll in enumerate(lstIdxSplit):
        print ('[%d/%d] process batch size [%d]' % (ii, numSplit, len(ll)))
        dataX, dataY = batcherVal.getBatchDataByIdx(parBatchIdx=ll)
        tret = modelTrained.predict_on_batch(dataX)
        # convert to 3D-data
        if batcherVal.isTheanoShape:
            tshape3D = list(dataX.shape[2:])
            tret3D = tret.reshape([dataX.shape[0]] + tshape3D + [tret.shape[-1]])
            dataY3D = dataY.reshape([dataX.shape[0]] + tshape3D + [tret.shape[-1]])
        else:
            tret3D = tret
            dataY3D = dataY
        sizSplit = len(ll)
        for iidx,idx in enumerate(ll):
            tpathMsk = batcherVal.arrPathDataMsk[idx]
            tmsk = nib.load(tpathMsk)
            # tsegm = nib.Nifti1Image(tret3D[iidx,:,:,:,1].copy().astype(tmsk.get_data_dtype()), tmsk.affine, header=tmsk.header)
            tsegm = nib.Nifti1Image(tret3D[iidx, :, :, :, 1].copy().astype(np.float16), tmsk.affine, header=tmsk.header)
            foutMsk = '%s-lungmask.nii.gz' % tpathMsk
            # tsegm.to_filename(foutMsk)
            nib.save(tsegm, foutMsk)
            print ('\t[%d/%d] * processing : %s --> %s' % (iidx, sizSplit, os.path.basename(tpathMsk), os.path.basename(foutMsk)))
    dt = time.time() - t0
    print (':: Inference time for #%d Samples (batch=%d) is %0.3fs, T/Sample=%0.3fs' % (batcherVal.numImg, parBatchSizeVal, dt, dt/batcherVal.numImg))