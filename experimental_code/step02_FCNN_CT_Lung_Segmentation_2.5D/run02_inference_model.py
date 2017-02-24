#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar (Alexander Kalinovsky)'

import os
import sys
import time

import keras
from keras import backend as K
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot as kplot

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

import nibabel as nib

from run00_common import BatcherOnImageCT3D, split_list_by_blocks

#############################################################
if __name__ == '__main__':
    #
    if K.image_dim_ordering() == 'tf':
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.86
        set_session(tf.Session(config=config))
    #
    parNumSlices   = 1
    parNumEpoch    = 30
    parBatchNumImg = 8
    parBatchNumSlc = 8
    parOptimizer = 'adam'
    # (1.1) UsSampled2D
    parModelType = 'InterpolatedUpSampling2D'
    # (1.2) Interpolated2D:
    # parModelType = 'InterpolatedUpSampling2D'
    # (1.3) Deconvoluted2D:
    # parModelType = 'Deconvolution2D'
    # fidxTrn = '/mnt/data1T/datasets/CRDF/CT_with_segm_mask_v3/resize-256x256x64/idx.txt'
    # fidxVal = '/mnt/data1T/datasets/CRDF/CT_with_segm_mask_v3/resize-256x256x64/idx.txt'
    fidxTrn = '../../experimental_data/resize-256x256x64/idx.txt'
    fidxVal = '../../experimental_data/resize-256x256x64/idx.txt'
    # (0) Basic configs
    parBatchNumImgTrn = parBatchNumImg
    parBatchNumImgVal = parBatchNumImg
    parBatchNumSlcTrn = parBatchNumSlc
    parBatchNumSlcVal = 32
    #
    parBatchSizeTrn = parBatchNumImgTrn*parBatchNumSlcTrn
    parBatchSizeVal = parBatchNumImgVal*parBatchNumSlcVal
    parIsLoadDataInMemory = False
    # (1) Load data into Batchers
    parIsTheanoShape = (K.image_dim_ordering() == 'th')
    batcherTrn = BatcherOnImageCT3D(fidxTrn, isTheanoShape=parIsTheanoShape, isLoadIntoMemory=False, numSlices=parNumSlices)
    # batcherVal = BatcherOnImageCT3D(fidxVal, isTheanoShape=parIsTheanoShape,
    #                                 pathMeanData=batcherTrn.pathMeanData,
    #                                 isLoadIntoMemory=False,
    #                                 numSlices=parNumSlices)
    batcherVal = batcherTrn #FIXME: disable separate validation on prod-training!
    print (batcherTrn)
    print (batcherVal)
    # (2) Configure training process
    parExportInfo = 'opt.%s.%s-slc%d' % (parOptimizer, parModelType, parNumSlices)
    numSlicesProc = (batcherTrn.sizeZ - 2*batcherTrn.numSlices)
    # (2.1) Precalculate approximate number of Iterations per Epoch
    parNumIterPerEpochTrn = (batcherTrn.numImg * numSlicesProc) / parBatchSizeTrn
    parNumIterPerEpochVal = (batcherVal.numImg * numSlicesProc) / parBatchSizeVal
    stepPrintVal = int(parNumIterPerEpochVal / 5)
    #
    model = batcherTrn.loadModelFromDir(pathDirWithModels=batcherTrn.wdir,  paramFilter=parExportInfo)
    # model.compile(loss='categorical_crossentropy', optimizer=parOptimizer, metrics=['accuracy'])
    # model.summary()
    for ii in range(batcherVal.numImg):
        tpathMsk  = batcherVal.arrPathDataMsk[ii]
        if K.image_dim_ordering()=='th':
            raise NotImplementedError
        else:
            tsegm3D   = np.zeros(batcherVal.shapeImg[:-1], np.float)
        lstIdxScl = range(batcherVal.numSlices, batcherVal.sizeZ-batcherVal.numSlices)
        lstIdxScl = split_list_by_blocks(lstIdxScl, parBatchNumSlcVal)
        for ss, sslst in enumerate(lstIdxScl):
            dataX, dataY = batcherVal.getBatchDataSlicedByIdx({
                ii: sslst
            }, isReturnDict=False)
            tret = model.predict_on_batch(dataX)
            if K.image_dim_ordering()=='th':
                raise NotImplementedError
            else:
                sizXY = batcherVal.shapeImg[:2]
                tret  = tret.transpose((1,0,2))
                tret  = tret.reshape(list(sizXY) + list(tret.shape[1:]))
                tmskSlc = (tret[:,:,:,1]>0.5).astype(np.float)
                tsegm3D[:,:,sslst] = tmskSlc
        tmsk = nib.load(tpathMsk)
        tsegm = nib.Nifti1Image(tsegm3D.copy().astype(np.float16), tmsk.affine, header=tmsk.header)
        foutMsk = '%s-segm-%s-slc%d.nii.gz' % (tpathMsk, parModelType, parNumSlices)
        nib.save(tsegm, foutMsk)
        print ('\t[%d/%d] * processing : %s --> %s' % (ii, batcherVal.numImg, os.path.basename(tpathMsk), os.path.basename(foutMsk)))
        # tmsk = nib.load(tpathMsk)
