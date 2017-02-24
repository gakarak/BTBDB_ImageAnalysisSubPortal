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

from keras.models import Model
from keras.layers import InputLayer, Convolution2D, \
    MaxPooling2D, UpSampling2D, Activation, \
    Input, Reshape, Permute, Deconvolution2D

from run00_common import BatcherOnImageCT3D, UpSamplingInterpolated2D, split_list_by_blocks

#############################################################
def buildModelSegNet_UpSampling2D(inpShape=(1, 256, 256), numCls=2, kernelSize=3):
    dataInput = Input(shape=inpShape)
    # -------- Encoder --------
    # Conv #1
    x = Convolution2D(nb_filter=16, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(dataInput)
    x = MaxPooling2D(pool_size=(2,2))(x)
    # Conv #2
    x = Convolution2D(nb_filter=32, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Conv #3
    x = Convolution2D(nb_filter=64, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Conv #4
    x = Convolution2D(nb_filter=128, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    # -------- Decoder --------
    # UpConv #1
    x = Convolution2D(nb_filter=128, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    # UpConv #2
    x = Convolution2D(nb_filter=64, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    # UpConv #3
    x = Convolution2D(nb_filter=32, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    retModel = Model(dataInput, x)
    # UpConv #4
    x = Convolution2D(nb_filter=16, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    #
    # 1x1 Convolution: emulation of Dense layer
    x = Convolution2D(nb_filter=numCls, nb_row=1, nb_col=1,
                      border_mode='valid', activation='linear')(x)
    tmpModel = Model(dataInput, x)
    if K.image_dim_ordering() == 'th':
        tmpShape = tmpModel.output_shape[-2:]
        sizeReshape = np.prod(tmpShape)
        x = Reshape([numCls, sizeReshape])(x)
        x = Permute((2, 1))(x)
    else:
        tmpShape = tmpModel.output_shape[1:-1]
        sizeReshape = np.prod(tmpShape)
        x = Reshape([sizeReshape, numCls])(x)
    x = Activation('softmax')(x)
    retModel = Model(dataInput, x)
    return retModel

#############################################################
def buildModelSegNet_InterpolatedUpSampling2D(inpShape=(1, 256, 256), numCls=2, kernelSize=3, order=1):
    dataInput = Input(shape=inpShape)
    # -------- Encoder --------
    # Conv #1
    x = Convolution2D(nb_filter=16, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(dataInput)
    x = MaxPooling2D(pool_size=(2,2))(x)
    # Conv #2
    x = Convolution2D(nb_filter=32, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Conv #3
    x = Convolution2D(nb_filter=64, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Conv #4
    x = Convolution2D(nb_filter=128, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    # -------- Decoder --------
    # UpConv #1
    x = Convolution2D(nb_filter=128, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = UpSamplingInterpolated2D(size=(2, 2), order=order)(x)
    # UpConv #2
    x = Convolution2D(nb_filter=64, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = UpSamplingInterpolated2D(size=(2, 2), order=order)(x)
    # UpConv #3
    x = Convolution2D(nb_filter=32, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = UpSamplingInterpolated2D(size=(2, 2), order=order)(x)
    retModel = Model(dataInput, x)
    # UpConv #4
    x = Convolution2D(nb_filter=16, nb_row=kernelSize, nb_col=kernelSize,
                      border_mode='same', activation='relu')(x)
    x = UpSamplingInterpolated2D(size=(2, 2), order=order)(x)
    #
    # 1x1 Convolution: emulation of Dense layer
    x = Convolution2D(nb_filter=numCls, nb_row=1, nb_col=1,
                      border_mode='valid', activation='linear')(x)
    tmpModel = Model(dataInput, x)
    if K.image_dim_ordering()=='th':
        tmpShape = tmpModel.output_shape[-2:]
        sizeReshape = np.prod(tmpShape)
        x = Reshape([numCls, sizeReshape])(x)
        x = Permute((2,1))(x)
    else:
        tmpShape = tmpModel.output_shape[1:-1]
        sizeReshape = np.prod(tmpShape)
        x = Reshape([sizeReshape,numCls])(x)
    x = Activation('softmax')(x)
    retModel = Model(dataInput, x)
    return retModel

#############################################################
if __name__ == '__main__':
    #
    if K.image_dim_ordering() == 'tf':
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.86
        set_session(tf.Session(config=config))
    #
    if len(sys.argv) > 8:
        parNumEpoch     = int(sys.argv[1])
        parBatchNumImg  = int(sys.argv[2])
        parBatchNumSlc  = int(sys.argv[3])
        parOptimizer    = sys.argv[4]
        parModelType    = sys.argv[5]#'UpSampling2D'
        parNumSlices    = int(sys.argv[6])
        fidxTrn         = sys.argv[7]
        fidxVal         = sys.argv[8]
    elif len(sys.argv) > 1:
        parNumEpoch    = 30
        parBatchNumImg = 8
        parBatchNumSlc = 8
        parOptimizer = 'adam'
        # (1) UsSampled2D
        # parModelType = 'UpSampling2D'
        parNumSlices = 1
        parModelType = 'InterpolatedUpSampling2D'
        # (2) Interpolated2D:
        # parModelType = 'InterpolatedUpSampling2D'
        # (3) Deconvoluted2D:
        # parModelType = 'Deconvolution2D'
        fidxTrn = '/mnt/data1T/datasets/CRDF/CT_with_segm_mask_v3/resize-256x256x64/idx.txt'
        fidxVal = '/mnt/data1T/datasets/CRDF/CT_with_segm_mask_v3/resize-256x256x64/idx.txt'
    else:
        print ('Usage: %s {#epoch} {#ImgPerBatch} {#SlicesPerBatch}'
               '\n\t{optimizer:adam}'
               '\n\t{modelType: UpSampling2D|InterpolatedUpSampling2D|Deconvolution2D}'
               '\n\t{#Slices per 3D crop}'
               '\n\t{/path/to/train-idx.txt} {/path/to/val-idx.txt}')
        sys.exit(0)
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
    batcherTrn = BatcherOnImageCT3D(fidxTrn, isTheanoShape=parIsTheanoShape, isLoadIntoMemory=True, numSlices=parNumSlices)
    # batcherVal = BatcherOnImageCT3D(fidxVal, isTheanoShape=parIsTheanoShape,
    #                                 pathMeanData=batcherTrn.pathMeanData,
    #                                 isLoadIntoMemory=False,
    #                                 numSlices=parNumSlices)
    batcherVal = batcherTrn  #FIXME: disable separate validation on prod-training!
    if K.image_dim_ordering() == 'tf':
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        set_session(tf.Session(config=config))
    print (batcherTrn)
    print (batcherVal)
    # (2) Configure training process
    parExportInfo = 'opt.%s.%s-slc%d' % (parOptimizer, parModelType, parNumSlices)
    numSlicesProc = (batcherTrn.sizeZ - 2*batcherTrn.numSlices)
    # (2.1) Precalculate approximate number of Iterations per Epoch
    parNumIterPerEpochTrn = (batcherTrn.numImg * numSlicesProc) / parBatchSizeTrn
    parNumIterPerEpochVal = (batcherVal.numImg * numSlicesProc) / parBatchSizeVal
    stepPrintTrn = int(parNumIterPerEpochTrn / 5)
    stepPrintVal = int(parNumIterPerEpochVal / 5)
    if stepPrintTrn < 1:
        stepPrintTrn = parNumIterPerEpochTrn
    if stepPrintVal < 1:
        stepPrintVal = parNumIterPerEpochVal
    #
    pcfg = {
        'opt':      parOptimizer,
        'mtype':    parModelType,
        'nslices':  parNumSlices
    }
    print ('*** Train params *** : #Epoch=%d, #BatchSize=%d, #IterPerEpoch=%d, %s'
           % (parNumEpoch, parBatchSizeTrn, parNumIterPerEpochTrn, pcfg))
    # (3) Build & visualize model
    if parModelType == 'UpSampling2D':
        model = buildModelSegNet_UpSampling2D(inpShape=batcherTrn.shapeImgSlc, numCls=batcherTrn.numCls)
    elif parModelType == 'InterpolatedUpSampling2D':
        model = buildModelSegNet_InterpolatedUpSampling2D(inpShape=batcherTrn.shapeImgSlc, numCls=batcherTrn.numCls)
    else:
        raise Exception('Unknown model type: [%s]' % parModelType)
    #
    model.compile(loss='categorical_crossentropy', optimizer=parOptimizer, metrics=['accuracy'])
    model.summary()
    # (4) Train model
    t0 = time.time()
    for eei in range(parNumEpoch):
        print ('[TRAIN] Epoch [%d/%d]' % (eei, parNumEpoch))
        # (1) train step
        tmpT1 = time.time()
        for ii in range(parNumIterPerEpochTrn):
            dataX, dataY = batcherTrn.getBatchDataSliced(parNumImages=parBatchNumImgTrn, parNumSlices=parBatchNumSlcTrn, isReturnDict=False)
            tret = model.train_on_batch(dataX, dataY)
            if (ii % stepPrintTrn) == 0:
                print ('\t[train] epoch [%d/%d], iter = [%d/%d] : loss[iter]/acc[iter] = %0.3f/%0.2f%%'
                       % (eei, parNumEpoch, ii, parNumIterPerEpochTrn, tret[0], 100. * tret[1]))
        tmpDT = time.time() - tmpT1
        print ('\t*** train-time for epoch #%d is %0.2fs' % (eei, tmpDT))
        # (2) model validation step
        if ((eei + 1) % 3) == 0:
            tmpIdxListSclices = split_list_by_blocks(
                range(batcherTrn.numSlices, batcherTrn.sizeZ - batcherTrn.numSlices), parBatchNumSlcVal)
            tmpIdxList = range(batcherVal.numImg)
            # lstIdxSplit = split_list_by_blocks(tmpIdxList, parBatchSizeVal)
            tmpVal = []
            for imgIdx in tmpIdxList:
                for tlistSliceIdx in tmpIdxListSclices:
                    dataX, dataY = batcherTrn.getBatchDataSlicedByIdx(dictImg2SliceIdx={imgIdx: tlistSliceIdx},
                                                                      isReturnDict=False)
                    tret = model.evaluate(dataX, dataY, verbose=False)
                    tmpVal.append(tret)
                if (imgIdx % stepPrintVal) == 0:
                    print ('\t\t[val] epoch [%d/%d], iter = [%d/%d] : loss[iter]/acc[iter] = %0.3f/%0.2f%%'
                           % (eei, parNumEpoch, imgIdx, parNumIterPerEpochVal, tret[0], 100. * tret[1]))
            tmpVal = np.array(tmpVal)
            tmeanValLoss = np.mean(tmpVal[:, 0])
            tmeanValAcc = np.mean(tmpVal[:, 1])
            print ('\t::validation: mean-losss/mean-acc = %0.3f/%0.3f' % (tmeanValLoss, tmeanValAcc))
        # (3) export model step
        if ((eei + 1) % 3) == 0:
            tmpT1 = time.time()
            tmpFoutModel = batcherTrn.exportModel(model, eei + 1, extInfo=parExportInfo)
            tmpDT = time.time() - tmpT1
            print ('[EXPORT] Epoch [%d/%d], export to [%s], time is %0.3fs' % (eei, parNumEpoch, tmpFoutModel, tmpDT))
    dt = time.time() - t0
    print ('Time for #%d Epochs is %0.3fs, T/Epoch=%0.3fs' % (parNumEpoch, dt, dt / parNumEpoch))
