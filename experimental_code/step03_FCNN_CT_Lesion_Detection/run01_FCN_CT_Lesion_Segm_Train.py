#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import sys
import time
import numpy as np
import json

import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skolor
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution3D, Activation, MaxPooling3D,\
    Flatten, BatchNormalization, InputLayer, Dropout, Reshape, Permute, Input, UpSampling3D, Lambda
from keras.layers.normalization import BatchNormalization

try:
   import cPickle as pickle
except:
   import pickle

import keras.optimizers as opt
from keras.utils.visualize_util import plot as kplot

from run00_common import BatcherOnImageCT3D, split_list_by_blocks

######################################################
def buildModel_CT(inpShape=(1, 128, 128, 64), numCls=2, sizFlt=3):
    dataInput = Input(shape=inpShape)
    # -------- Encoder --------
    # Conv1
    x = Convolution3D(nb_filter=16, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(dataInput)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    # Conv2
    # x = Convolution3D(nb_filter=32, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
    #                   border_mode='same', activation='relu')(x)
    x = Convolution3D(nb_filter=32, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    # Conv3
    # x = Convolution3D(nb_filter=64, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
    #                   border_mode='same', activation='relu')(x)
    x = Convolution3D(nb_filter=64, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    # Conv4
    # x = Convolution3D(nb_filter=128, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
    #                   border_mode='same', activation='relu')(x)
    x = Convolution3D(nb_filter=128, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    # Conv5
    # x = Convolution3D(nb_filter=256, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    # x = MaxPooling3D(pool_size=(2, 2))(x)
    # Conv6
    # x = Convolution3D(nb_filter=256, nb_col=sizFlt, nb_row=sizFlt, border_mode='same', activation='relu')(x)
    # x = MaxPooling3D(pool_size=(2, 2))(x)
    # -------- Decoder --------
    # UpConv #1
    x = Convolution3D(nb_filter=128, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    # UpConv #2
    x = Convolution3D(nb_filter=64, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    # UpConv #3
    x = Convolution3D(nb_filter=32, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    # UpConv #4
    x = Convolution3D(nb_filter=16, kernel_dim1=sizFlt, kernel_dim2=sizFlt, kernel_dim3=sizFlt,
                      border_mode='same', activation='relu')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    # 1x1 Convolution: emulation of Dense layer
    x = Convolution3D(nb_filter=numCls, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                      border_mode='valid', activation='linear')(x)
    # -------- Finalize --------
    #
    tmpModel = Model(dataInput, x)
    if K.image_dim_ordering()=='th':
        tmpShape = tmpModel.output_shape[-3:]
        sizeReshape = np.prod(tmpShape)
        x = Reshape([numCls, sizeReshape])(x)
        x = Permute((2, 1))(x)
        x = Activation('softmax')(x)
        retModel = Model(dataInput, x)
    else:
        # x = Lambda(lambda XX: tf.nn.softmax(XX))(x)
        tmpShape = tmpModel.output_shape[1:-1]
        sizeReshape = np.prod(tmpShape)
        x = Reshape([sizeReshape, numCls])(x)
        retModel = Model(dataInput, x)
    retShape = retModel.output_shape[1:-1]
    return (retModel, retShape)


######################################################
def usage(pargv):
    print ('Usage: %s {/path/to/train-idx.txt} {/path/to/validation-idx.txt}' % pargv[0])

######################################################
if __name__=='__main__':
    if len(sys.argv)>5:
        parNumEpoch  = int(sys.argv[1])
        parBatchSize = int(sys.argv[2])
        parOptimizer = sys.argv[3]
        fidxTrain    = sys.argv[4]
        fidxVal      = sys.argv[5]
    else:
        parOptimizer = 'adam'
        parNumEpoch  = 100
        parBatchSize = 8
        fidxTrain    = '/mnt/data1T/datasets/CRDF/TB_5_Classes/TB_sub_1_5-resize-128x128x64/idx.txt-train.txt'
        fidxVal      = '/mnt/data1T/datasets/CRDF/TB_5_Classes/TB_sub_1_5-resize-128x128x64/idx.txt-val.txt'
    if not os.path.isfile(fidxTrain):
        usage(sys.argv)
        raise Exception('Cant find Train-Index path: [%s]' % fidxTrain)
    if not os.path.isfile(fidxVal):
        usage(sys.argv)
        raise Exception('Cant find Validation-Index path: [%s]' % fidxVal)
    #
    parBatchSizeTrain = parBatchSize
    parBatchSizeVal   = parBatchSize
    # parIsTheanoShape  = True
    parIsTheanoShape = (K.image_dim_ordering()=='th')
    parIsLoadIntoMemory = True
    parClassWeights   = [1., 12.]
    # parClassWeights   = None
    batcherTrain = BatcherOnImageCT3D(pathDataIdx=fidxTrain,
                                      isTheanoShape=parIsTheanoShape, isLoadIntoMemory=parIsLoadIntoMemory)
    batcherVal   = BatcherOnImageCT3D(pathDataIdx=fidxVal,
                                      pathMeanData=batcherTrain.pathMeanData,
                                      isTheanoShape=parIsTheanoShape, isLoadIntoMemory=parIsLoadIntoMemory)
    print (':: Train data: %s' % batcherTrain)
    print (':: Val   data: %s' % batcherVal)
    #
    parNumIterPerEpochTrain = batcherTrain.getNumImg() / parBatchSizeTrain
    parNumIterPerEpochVal   = batcherVal.getNumImg() / parBatchSizeTrain
    stepPrintTrain = int(parNumIterPerEpochTrain / 5)
    stepPrintVal = int(parNumIterPerEpochVal / 5)
    if stepPrintTrain < 1:
        stepPrintTrain = parNumIterPerEpochTrain
    if stepPrintVal < 1:
        stepPrintVal = parNumIterPerEpochVal
    #
    parInputShape = batcherTrain.shapeImg
    model,_ = buildModel_CT(inpShape=parInputShape)
    model.compile(loss='categorical_crossentropy',
                  optimizer=parOptimizer,
                  # optimizer=opt.SGD(lr=0.01, momentum=0.8, nesterov=True),
                  metrics=['accuracy'])
    model.summary()
    fimgModel = 'ct-segnet-model-tf.jpg'
    kplot(model, fimgModel, show_shapes=True)
    # plt.imshow(skio.imread(fimgModel))
    # plt.title(batcherTrain.shapeImg)
    # plt.show(block=False)
    #
    t0 = time.time()
    for eei in range(parNumEpoch):
        print ('[TRAIN] Epoch [%d/%d]' % (eei, parNumEpoch))
        # (0) prepare params
        tmpT1 = time.time()
        # (1) model train step
        for ii in range(parNumIterPerEpochTrain):
            dataX, dataY = batcherTrain.getBatchData(parBatchSize=parBatchSizeTrain)
            tret = model.train_on_batch(dataX, dataY, class_weight=parClassWeights)
            if (ii % stepPrintTrain) == 0:
                print ('\t[train] epoch [%d/%d], iter = [%d/%d] : loss[iter]/acc[iter] = %0.3f/%0.2f%%'
                       % (eei, parNumEpoch, ii, parNumIterPerEpochTrain, tret[0], 100. * tret[1]))
        tmpDT = time.time() - tmpT1
        print ('\t*** train-time for epoch #%d is %0.2fs' % (eei, tmpDT))
        # (2) model validation model step
        if ((eei + 1) % 5) == 0:
            print ('[VALIDATION] Epoch [%d/%d]' % (eei, parNumEpoch))
            lstRanges = split_list_by_blocks(range(batcherVal.numImg), parBatchSizeVal)
            tmpVal = []
            for ii, ll in enumerate(lstRanges):
                dataX, dataY = batcherVal.getBatchDataByIdx(parBatchIdx=ll)
                tret = model.evaluate(dataX, dataY, verbose=False)
                tmpVal.append(tret)
                if (ii % stepPrintVal) == 0:
                    print ('\t\t[val] epoch [%d/%d], iter = [%d/%d] : loss[iter]/acc[iter] = %0.3f/%0.2f%%'
                           % (eei, parNumEpoch, ii, parNumIterPerEpochVal, tret[0], 100. * tret[1]))
            tmpVal = np.array(tmpVal)
            tmeanValLoss = np.mean(tmpVal[:, 0])
            tmeanValAcc  = np.mean(tmpVal[:, 1])
            print ('\t::validation: mean-losss/mean-acc = %0.3f/%0.3f' % (tmeanValLoss, tmeanValAcc))
        # (3) export model step
        if ((eei + 1) % 5) == 0:
            tmpT1 = time.time()
            tmpFoutModel = batcherTrain.exportModel(model, eei + 1, extInfo='opt.%s' % parOptimizer)
            tmpDT = time.time() - tmpT1
            print ('[EXPORT] Epoch [%d/%d], export to [%s], time is %0.3fs' % (eei, parNumEpoch, tmpFoutModel, tmpDT))
    dt = time.time() - t0
    print ('Time for #%d Epochs is %0.3fs, T/Epoch=%0.3fs' % (parNumEpoch, dt, dt / parNumEpoch))

