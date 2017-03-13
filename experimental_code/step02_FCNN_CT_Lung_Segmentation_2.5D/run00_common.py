#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar (Alexander Kalinovsky)'

import glob
import os
import sys
import time
import numpy as np
import json

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

import nibabel as nib

try:
   import cPickle as pickle
except:
   import pickle

import collections
# from collections import OrderedDict
import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skolor
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution3D, Activation, MaxPooling3D,\
    Flatten, BatchNormalization, InputLayer, Dropout, Reshape, Permute, Input, UpSampling3D, Lambda
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils.visualize_util import plot as kplot

######################################################
def resize_images_interpolated(X, height_factor, width_factor, order, dim_ordering):
    """
    Simple modification of the original Keras code (by ar)
    ****
    Resizes the images contained in a 4D tensor of shape
    - `[batch, channels, height, width]` (for 'th' dim_ordering)
    - `[batch, height, width, channels]` (for 'tf' dim_ordering)
    by a factor of `(height_factor, width_factor)`. Both factors should be
    positive integers.

    # Returns
        A tensor.
    """
    resizeMethod = 0
    if order==0:
        resizeMethod = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif order==1:
        resizeMethod = tf.image.ResizeMethod.BILINEAR
    elif order==2:
        resizeMethod = tf.image.ResizeMethod.BICUBIC
    else:
        raise Exception('Incorrect interpolation method order [%s], currently available values is 0,1,2')
    if dim_ordering == 'th':
        original_shape = K.int_shape(X)
        new_shape = tf.shape(X)[2:]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = K.permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_images(X, new_shape, method=resizeMethod)
        # X = tf.image.resize_nearest_neighbor(X, new_shape)
        X = K.permute_dimensions(X, [0, 3, 1, 2])
        X.set_shape((None, None, original_shape[2] * height_factor if original_shape[2] is not None else None,
                     original_shape[3] * width_factor if original_shape[3] is not None else None))
        return X
    elif dim_ordering == 'tf':
        original_shape = K.int_shape(X)
        new_shape = tf.shape(X)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        # X = tf.image.resize_nearest_neighbor(X, new_shape)
        X = tf.image.resize_images(X, new_shape, method=resizeMethod)
        X.set_shape((None, original_shape[1] * height_factor if original_shape[1] is not None else None,
                     original_shape[2] * width_factor if original_shape[2] is not None else None, None))
        return X
    else:
        raise ValueError('Invalid dim_ordering:', dim_ordering)

######################################################
class UpSamplingInterpolated2D(Layer):

    def __init__(self, size=(2, 2), order=2, dim_ordering='default', **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.size = tuple(size)
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering must be in {tf, th}.')
        if dim_ordering != 'tf':
            raise Exception('Layer <UpSamplingInterpolated2D> currently supported only Tensorflow backend!')
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]
        self.order = order
        super(UpSamplingInterpolated2D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            width = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            height = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.dim_ordering == 'tf':
            width = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            height = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

    def call(self, x, mask=None):
        # return K.resize_images(x, self.size[0], self.size[1], self.dim_ordering)
        return resize_images_interpolated(x, self.size[0], self.size[1], self.order, self.dim_ordering)

    def get_config(self):
        config = {
            'size': self.size,
            'order': self.order
        }
        base_config = super(UpSamplingInterpolated2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

######################################################
def split_list_by_blocks(lst, psiz):
    tret = [lst[x:x + psiz] for x in range(0, len(lst), psiz)]
    return tret

######################################################
class BatcherOnImageCT3D:
    pathDataIdx=None
    pathMeanData=None
    meanPrefix='mean.pkl'
    arrPathDataImg=None
    arrPathDataMsk=None
    wdir=None
    dataImg     = None
    dataMsk     = None
    dataMskCls  = None
    meanData    = None
    #
    imgScale  = 1.
    modelPrefix = None
    #
    isTheanoShape=True
    # isRemoveMeanImage=False
    isDataInMemory=False
    shapeImg    = None
    shapeImgSlc = None
    shapeMsk    = None
    shapeMskSlc = None
    sizeZ = -1
    numCh = 1
    numImg = -1
    numSlices = -1
    #
    modelPath = None
    model = None
    def __init__(self, pathDataIdx=None, pathMeanData=None, numSlices=-1, isRecalculateMeanIfExist=False,
                 isTheanoShape=True,
                 isLoadIntoMemory=False):
        if pathDataIdx is not None:
            self.loadDataset(pathDataIdx=pathDataIdx,
                             pathMeanData=pathMeanData,
                             numSlices=numSlices,
                             isRecalculateMeanIfExist=isRecalculateMeanIfExist,
                             isTheanoShape=isTheanoShape,
                             isLoadIntoMemory=isLoadIntoMemory)
    def loadDataset(self, pathDataIdx, pathMeanData=None, numSlices=-1, isRecalculateMeanIfExist=False,
                    isTheanoShape=True,
                    isLoadIntoMemory=False):
        self.isTheanoShape=isTheanoShape
        # self.isRemoveMeanImage=isRemoveMeanImage
        # (1) Check input Image
        if not os.path.isfile(pathDataIdx):
            raise Exception('Cant find input Image file [%s]' % pathDataIdx)
        self.pathDataIdx = os.path.abspath(pathDataIdx)
        self.wdir = os.path.dirname(self.pathDataIdx)
        tdata = pd.read_csv(self.pathDataIdx)
        # (2) Check input Image Mask
        # self.pathDataMsk = '%s_msk.png' % os.path.splitext(self.pathDataImg)[0]
        self.arrPathDataImg = np.array([os.path.join(self.wdir, xx) for xx in tdata['path']])
        self.arrPathDataMsk = np.array([os.path.join(self.wdir, xx) for xx in tdata['pathmsk']])
        # (3) Load Image and Mask
        tpathImg = self.arrPathDataImg[0]
        tpathMsk = self.arrPathDataMsk[0]
        if not os.path.isfile(tpathImg):
            raise Exception('Cant find CT Image file [%s]' % tpathImg)
        if not os.path.isfile(tpathMsk):
            raise Exception('Cant find CT Image Mask file [%s]' % tpathMsk)
        tdataImg = nib.load(tpathImg).get_data()
        tdataMsk = nib.load(tpathMsk).get_data()
        tdataImg = self.adjustImage(self.transformImageFromOriginal(tdataImg, isRemoveMean=False))
        tdataMsk = self.transformImageFromOriginal(tdataMsk>200, isRemoveMean=False)
        self.numCls = len(np.unique(tdataMsk))
        tdataMskCls = self.convertMskToOneHot(tdataMsk)
        self.shapeImg = tdataImg.shape
        self.shapeMsk = tdataMskCls.shape
        self.numSlices = numSlices
        if numSlices<0:
            self.shapeImgSlc = self.shapeImg
            self.shapeMskSlc = self.shapeMsk
        else:
            tnumSlc = 2*self.numSlices+1
            if K.image_dim_ordering()=='th':
                self.shapeImgSlc = list(self.shapeImg[:-1]) + [tnumSlc]
                self.shapeMskSlc = list(self.shapeMsk[:-1]) + [tnumSlc]
                self.sizeZ = self.shapeImg[-1]
            else:
                self.shapeImgSlc = list(self.shapeImg[:2]) + [tnumSlc]      #+ [self.shapeImg[-1]]
                self.shapeMskSlc = list(self.shapeMsk[:2]) + [self.numCls]  #+ [self.shapeMsk[-1]]
                self.sizeZ = self.shapeImg[-2]
        # (4) Check input Mean Image Data
        if pathMeanData is None:
            self.pathMeanData = '%s-%s' % (self.pathDataIdx, self.meanPrefix)
            self.precalculateAndLoadMean(isRecalculateMean=isRecalculateMeanIfExist)
        else:
            if not os.path.isfile(pathMeanData):
                raise Exception('Cant find MEAN-data file [%s]' % pathMeanData)
            self.pathMeanData = pathMeanData
            self.precalculateAndLoadMean(isRecalculateMean=isRecalculateMeanIfExist)
        # (5) Load data into memory
        self.numImg = len(self.arrPathDataImg)
        if isLoadIntoMemory:
            #FIXME: incorrect code, please, fix this code before using
            self.isDataInMemory = True
            self.dataImg = np.zeros([self.numImg] + list(self.shapeImg), dtype=np.float)
            self.dataMsk = None
            # self.dataMsk = np.zeros([self.numImg] + list(self.shapeImg), dtype=np.float)
            self.dataMskCls = np.zeros([self.numImg] + list(self.shapeMsk), dtype=np.float)
            print (':: Loading data into memory:')
            for ii in range(self.numImg):
                tpathImg = self.arrPathDataImg[ii]
                tpathMsk = self.arrPathDataMsk[ii]
                #
                tdataImg = self.adjustImage(nib.load(tpathImg).get_data())
                tdataMsk = nib.load(tpathMsk).get_data()
                tdataImg = self.transformImageFromOriginal(tdataImg, isRemoveMean=True)
                tdataMsk = self.transformImageFromOriginal(tdataMsk > 200, isRemoveMean=False)
                tdataMskCls = self.convertMskToOneHot(tdataMsk)
                self.dataImg[ii] = tdataImg
                # self.dataMsk[ii] = tdataMsk
                self.dataMskCls[ii] = tdataMskCls
                if (ii % 10) == 0:
                    print ('\t[%d/%d] ...' % (ii, self.numImg))
            print ('\t... [done]')
            if self.isTheanoShape:
                tshp = self.dataMskCls.shape
                print (tshp)
        else:
            self.isDataInMemory = False
            self.dataImg    = None
            self.dataMsk    = None
            self.dataMskCls = None
    def getNumImg(self):
        if self.isInitialized():
            return self.numImg
        else:
            return 0
    def adjustImage(self, pimg):
        qmin = -1400.
        qmax = +400
        tret = pimg.copy()
        tret[pimg < qmin] = qmin
        tret[pimg > qmax] = qmax
        tret = (tret - qmin) / (qmax - qmin)
        return tret
    def convertMskToOneHot(self, msk):
        tshape = list(msk.shape)
        if self.numCls>2:
            tret = np_utils.to_categorical(msk.reshape(-1), self.numCls)
        else:
            tret = (msk.reshape(-1)>0).astype(np.float)
            tret = np.vstack((1.-tret,tret)).transpose()
        if self.isTheanoShape:
            tmpShape = list(tshape[1:]) + [self.numCls]
            # tshape[ 0] = self.numCls
        else:
            tmpShape = tshape
            tmpShape[-1] = self.numCls
        tret = tret.reshape(tmpShape)
        if self.isTheanoShape:
            #FIXME: work only for 3D!!!
            tret = tret.transpose((3,0,1,2))
        return tret
    def isInitialized(self):
        return (self.shapeImg is not None) and (self.shapeMsk is not None) and (self.wdir is not None) and (self.numCls>0)
    def checkIsInitialized(self):
        if not self.isInitialized():
            raise Exception('class Batcher() is not correctly initialized')
    # def toString(self):
    #     if self.isInitialized():
    #         tstr = 'Shape/Slice=%s/%s, #Samples=%d, #Labels=%d, #Slices=%s'\
    #                % (self.shapeImg, self.shapeImgSlc, self.numImg, self.numCls, self.numSlices)
    #         if self.meanData is not None:
    #             tstr = '%s, meanValuePerCh=%s' % (tstr, self.meanData['meanCh'])
    #         else:
    #             tstr = '%s, meanValuePerCh= is Not Calculated' % (tstr)
    #     else:
    #         tstr = "BatcherOnImageCT3D() is not initialized"
    #     return tstr
    def toString(self):
        if self.isInitialized():
            tstr = '#Samples=%d' % (self.numImg)
        else:
            tstr = "BatcherOnImage3D() is not initialized"
        # (1) number of classes
        if self.numCls is not None:
            tstr = '%s, #Cls=%d' % (tstr, self.numCls)
        # (2) input/output shapes
        tstr = '%s, InpShape=%s, OutShape=%s' % (tstr, self.shapeImg, self.shapeMsk)
        #
        if self.meanData is not None:
            tstr = '%s, meanValuePerCh=%s' % (tstr, self.meanData['meanCh'])
        else:
            tstr = '%s, meanValuePerCh= is Not Calculated' % (tstr)
        if (self.model is not None) and (self.modelPath is not None):
            tstr = '%s, model is loaded [%s]' % (tstr, os.path.basename(self.modelPath))
        return tstr
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def preprocImageShape(self, img):
        if self.isTheanoShape:
            return img.reshape([1] + list(img.shape))
        else:
            return img.reshape(list(img.shape) + [1])
    def removeMean(self, img):
        ret = img
        ret -= self.meanData['meanCh']
        # ret -= self.meanData['meanImg']
        return ret
    def transformImageFromOriginal(self, pimg, isRemoveMean=True):
        tmp = self.preprocImageShape(pimg)
        tmp = tmp.astype(np.float) / self.imgScale
        if isRemoveMean:
            tmp = self.removeMean(tmp)
        return tmp
    def precalculateAndLoadMean(self, isRecalculateMean=False):
        if os.path.isfile(self.pathMeanData) and (not isRecalculateMean):
            print (':: found mean-value file, try to load from it [%s] ...' % self.pathMeanData)
            with open(self.pathMeanData, 'r') as f:
                self.meanData = pickle.load(f)
            tmpMeanKeys = ('meanImg', 'meanCh', 'meanImgCh')
            for ii in tmpMeanKeys:
                if ii not in self.meanData.keys():
                    raise Exception('Mean-file is invalid. Cant find key-value in mean-file [%s]' % self.pathMeanData)
        else:
            self.meanData = {}
            self.meanData['meanImg'] = None
            self.meanData['meanImgCh'] = None
            maxNumImages = 1000
            if len(self.arrPathDataImg)<maxNumImages:
                maxNumImages = len(self.arrPathDataImg)
            rndIdx = np.random.permutation(range(len(self.arrPathDataImg)))[:maxNumImages]
            print ('*** Precalculate mean-info:')
            for ii,idx in enumerate(rndIdx):
                tpathImg = self.arrPathDataImg[idx]
                tdataImg = nib.load(tpathImg).get_data()
                tdataImg = self.adjustImage(self.transformImageFromOriginal(tdataImg, isRemoveMean=False))
                if self.meanData['meanImg'] is None:
                    self.meanData['meanImg'] = tdataImg
                else:
                    self.meanData['meanImg'] += tdataImg
                if (ii%10)==0:
                    print ('\t[%d/%d] ...' % (ii, len(rndIdx)))
            self.meanData['meanImg'] /= len(rndIdx)
            self.meanData['meanCh'] = np.mean(self.meanData['meanImg'])
            print (':: mean-image %s mean channels value is [%s], saved to [%s]'
                   % (self.meanData['meanImg'].shape, self.meanData['meanCh'], self.pathMeanData))
            with open(self.pathMeanData, 'wb') as f:
                pickle.dump(self.meanData, f)
    def getSlice25D(self, idxImg, zidx):
        if self.isDataInMemory:
            tnumSlices = len(zidx)
            dataX = np.zeros([tnumSlices] + list(self.shapeImgSlc), dtype=np.float)
            # dataM = np.zeros([tnumSlices] + list(self.shapeMskSlc), dtype=np.float)
            dataM = None
            dataY = None
            tnumBrd = self.numSlices + 1
            for ii, tidx in enumerate(zidx):
                if K.image_dim_ordering() == 'th':
                    # 3D-version
                    # dataX[ii] = tdataImg[:, :, :, tidx - tnumBrd + 1:tidx + tnumBrd ]
                    # dataY[ii] = tdataMskCls[:, :, :, tidx:tidx+1]
                    # 2D-version
                    dataX[ii] = self.dataImg[idxImg, 0, :, :, tidx - tnumBrd + 1:tidx + tnumBrd]
                    # dataM[ii] = self.dataMskCls[idxImg, :, :, :, tidx]
                else:
                    # 3D-version
                    # dataX[ii] = tdataImg[:, :, tidx - tnumBrd + 1:tidx + tnumBrd, :]
                    # dataY[ii] = tdataMskCls[:, :, tidx:tidx+1, :]
                    # 2D-version
                    timg = self.dataImg[idxImg, :, :, tidx - tnumBrd + 1:tidx + tnumBrd, 0]
                    tmsk = self.dataMskCls[idxImg, :, :, tidx, :]
                    tout = tmsk.reshape(-1, self.numCls)
                    if dataY is None:
                        dataY = np.zeros([tnumSlices] + list(tout.shape), dtype=np.float)
                    dataX[ii] = timg
                    dataY[ii] = tout
                    # dataM[ii] = tmsk
        else:
            tpathImg = self.arrPathDataImg[idxImg]
            tpathMsk = self.arrPathDataMsk[idxImg]
            tdataImg = self.adjustImage(nib.load(tpathImg).get_data())
            tdataMsk = nib.load(tpathMsk).get_data()
            tdataImg = self.transformImageFromOriginal(tdataImg, isRemoveMean=True)
            tdataMsk = self.transformImageFromOriginal(tdataMsk > 200, isRemoveMean=False)
            tdataMskCls = self.convertMskToOneHot(tdataMsk)
            #
            tnumSlices = len(zidx)
            dataX = np.zeros([tnumSlices] + list(self.shapeImgSlc), dtype=np.float)
            dataM = np.zeros([tnumSlices] + list(self.shapeMskSlc), dtype=np.float)
            dataY = None
            tnumBrd = self.numSlices+1
            for ii, tidx in enumerate(zidx):
                if K.image_dim_ordering()=='th':
                    # 3D-version
                    # dataX[ii] = tdataImg[:, :, :, tidx - tnumBrd + 1:tidx + tnumBrd ]
                    # dataY[ii] = tdataMskCls[:, :, :, tidx:tidx+1]
                    # 2D-version
                    dataX[ii] = tdataImg[0, :, :, tidx - tnumBrd + 1:tidx + tnumBrd ]
                    dataM[ii] = tdataMskCls[:, :, :, tidx]
                else:
                    # 3D-version
                    # dataX[ii] = tdataImg[:, :, tidx - tnumBrd + 1:tidx + tnumBrd, :]
                    # dataY[ii] = tdataMskCls[:, :, tidx:tidx+1, :]
                    # 2D-version
                    timg = tdataImg[:, :, tidx - tnumBrd + 1:tidx + tnumBrd, 0]
                    tmsk = tdataMskCls[:, :, tidx, :]
                    tout = tmsk.reshape(-1, self.numCls)
                    if dataY is None:
                        dataY = np.zeros([tnumSlices] + list(tout.shape), dtype=np.float)
                    dataX[ii] = timg
                    dataY[ii] = tout
                    dataM[ii] = tmsk
        return (dataX, dataY, dataM)
    def
    def getBatchDataSlicedByIdx(self, dictImg2SliceIdx, isReturnDict=True):
        dictDataX = collections.OrderedDict()
        dictDataY = collections.OrderedDict()
        for ii,(imgIdx, listIdxSlices) in enumerate(dictImg2SliceIdx.items()):
            tmpDataX, tmpDataY, _ = self.getSlice25D(imgIdx, listIdxSlices)
            dictDataX[imgIdx] = tmpDataX
            dictDataY[imgIdx] = tmpDataY
        if isReturnDict:
            return (dictDataX, dictDataY)
        else:
            return (np.concatenate(dictDataX.values()), np.concatenate(dictDataY.values()))
    def getBatchDataSliced(self, parNumImages=8, parNumSlices=4, isReturnDict=True):
        self.checkIsInitialized()
        numImg = self.numImg
        rndIdx = np.random.permutation(range(numImg))[:parNumImages]
        dictImg2Idx={}
        for imgIdx in rndIdx:
            trndSliceIdx = range(self.numSlices, self.sizeZ-self.numSlices)
            trndSliceIdx = np.random.permutation(trndSliceIdx)[:parNumSlices]
            dictImg2Idx[imgIdx] = trndSliceIdx
        return self.getBatchDataSlicedByIdx(dictImg2SliceIdx=dictImg2Idx, isReturnDict=isReturnDict)
    def getBatchDataByIdx(self, parBatchIdx):
        rndIdx = parBatchIdx
        parBatchSize = len(rndIdx)
        dataX = np.zeros([parBatchSize] + list(self.shapeImg), dtype=np.float)
        dataY = np.zeros([parBatchSize] + list(self.shapeMsk), dtype=np.float)
        for ii, tidx in enumerate(rndIdx):
            if self.isDataInMemory:
                dataX[ii] = self.dataImg[tidx]
                dataY[ii] = self.dataMskCls[tidx]
            else:
                tpathImg = self.arrPathDataImg[tidx]
                tpathMsk = self.arrPathDataMsk[tidx]
                tdataImg = self.adjustImage(nib.load(tpathImg).get_data())
                tdataMsk = nib.load(tpathMsk).get_data()
                tdataImg = self.transformImageFromOriginal(tdataImg, isRemoveMean=True)
                tdataMsk = self.transformImageFromOriginal(tdataMsk > 200, isRemoveMean=False)
                tdataMskCls = self.convertMskToOneHot(tdataMsk)
                dataX[ii] = tdataImg
                dataY[ii] = tdataMskCls
        if self.isTheanoShape:
            tshp = dataY.shape
            dataY = dataY.reshape([tshp[0], tshp[1], np.prod(tshp[-3:])]).transpose((0, 2, 1))
            # print (tshp)
        return (dataX, dataY)
    def getBatchData(self, parBatchSize=8):
        self.checkIsInitialized()
        numImg = self.numImg
        rndIdx = np.random.permutation(range(numImg))[:parBatchSize]
        return self.getBatchDataByIdx(rndIdx)
    def exportModel(self, model, epochId, extInfo=None):
        if extInfo is not None:
            modelPrefix = extInfo
        else:
            modelPrefix = ''
        foutModel = "%s-e%03d.json" % (modelPrefix, epochId)
        foutWeights = "%s-e%03d.h5" % (modelPrefix, epochId)
        foutModel = '%s-%s' % (self.pathDataIdx, foutModel)
        foutWeights = '%s-%s' % (self.pathDataIdx, foutWeights)
        with open(foutModel, 'w') as f:
            str = json.dumps(json.loads(model.to_json()), indent=3)
            f.write(str)
        model.save_weights(foutWeights, overwrite=True)
        return foutModel
    @staticmethod
    def loadModelFromJson(pathModelJson):
        if not os.path.isfile(pathModelJson):
            raise Exception('Cant find JSON-file [%s]' % pathModelJson)
        tpathBase = os.path.splitext(pathModelJson)[0]
        tpathModelWeights = '%s.h5' % tpathBase
        if not os.path.isfile(tpathModelWeights):
            raise Exception('Cant find h5-Weights-file [%s]' % tpathModelWeights)
        with open(pathModelJson, 'r') as f:
            tmpStr = f.read()
            model = keras.models.model_from_json(tmpStr, custom_objects={'UpSamplingInterpolated2D': UpSamplingInterpolated2D})
            model.load_weights(tpathModelWeights)
        return model
    def loadModelFromDir(self, pathDirWithModels, paramFilter=None):
        if paramFilter is None:
            lstModels = glob.glob('%s/*.json' % pathDirWithModels)
        else:
            lstModels = glob.glob('%s/*%s*.json' % (pathDirWithModels, paramFilter))
        pathJson  = os.path.abspath(sorted(lstModels)[-1])
        print (':: found model [%s] in directory [%s]' % (os.path.basename(pathJson), pathDirWithModels))
        self.modelPath = pathJson
        return BatcherOnImageCT3D.loadModelFromJson(pathJson)
    def loadModelForInference(self, pathModelJson, pathMeanData, paramFilter=None):
        if os.path.isdir(pathModelJson):
            self.model = self.loadModelFromDir(pathModelJson)
        else:
            self.model = BatcherOnImageCT3D.loadModelFromJson(pathModelJson)
        if os.path.isdir(pathMeanData):
            if paramFilter is None:
                lstMean = sorted(glob.glob('%s/*mean.pkl' % pathMeanData))
            else:
                lstMean = sorted(glob.glob('%s/*%s*mean.pkl' % (pathMeanData, paramFilter)))
            if len(lstMean) < 1:
                raise Exception('Cant find mean-file in directory [%s]' % pathMeanData)
            self.pathMeanData = lstMean[0]
        else:
            self.pathMeanData = pathMeanData
        self.precalculateAndLoadMean(isRecalculateMean=False)
        self.numCls = self.model.output_shape[-1]
        self.shapeImgSlc = self.model.input_shape[1:]
        self.isTheanoShape = (K.image_dim_ordering() == 'th')
        if self.isTheanoShape:
            self.shapeMskSlc = tuple([self.numCls] + list(self.shapeImgSlc[1:]))
            self.numSlices = (self.model.input_shape[1] - 1) / 2
        else:
            self.shapeMskSlc = tuple(list(self.shapeImgSlc[:-1]) + [self.numCls])
            self.numSlices = (self.model.input_shape[-1] - 1) / 2
    def inference(self, lstData, batchSize=2):
        if self.model is None:
            raise Exception('Model is not loaded... load model before call inferece()')
        if len(lstData) > 0:
            tmpListOfImg = []
            # (1) load into memory
            if isinstance(lstData[0], str) or isinstance(lstData[0], unicode):
                for ii in lstData:
                    tmpListOfImg.append(nib.load(ii).get_data())
            else:
                tmpListOfImg = lstData
            # (2) check shapes
            tsetShapes = set()
            for ii in tmpListOfImg:
                tsetShapes.add(ii.shape[:-1])
            if len(tsetShapes) > 1:
                raise Exception('Shapes of images must be equal sized')
            tmpShape = self.shapeImgSlc[:-1]
            if tmpShape not in tsetShapes:
                raise Exception('Model input shape and shapes of input images is not equal!')
            # (3) convert data
            self.isDataInMemory = True
            numImg = len(tmpListOfImg)
            # self.dataImg = np.zeros([numImg] + list(self.shapeImg), dtype=np.float)
            for ii in range(numImg):
                tdataImg = self.adjustImage(tmpListOfImg[ii])
                tdataImg = self.transformImageFromOriginal(tdataImg, isRemoveMean=True)
                if K.image_dim_ordering() == 'th':
                    tsegm3D = np.zeros(tdataImg.shapeImg[+1:], np.float)
                    numSlicesZ = tdataImg.shape[-1]
                else:
                    tsegm3D = np.zeros(tdataImg.shape[:-1], np.float)
                    numSlicesZ = tdataImg.shape[-2]
                lstIdxScl = range(self.numSlices, numSlicesZ - self.numSlices)
                lstIdxScl = split_list_by_blocks(lstIdxScl, batchSize)
                for ss, sslst in enumerate(lstIdxScl):
                    dataX, dataY = self.getBatchDataSlicedByIdx({
                        ii: sslst
                    }, isReturnDict=False)
                    tret = self.model.predict_on_batch(dataX)
                    if K.image_dim_ordering() == 'th':
                        # raise NotImplementedError
                        sizXY = self.shapeImg[1:-1]

                    else:
                        sizXY = self.shapeImg[:2]
                        tret = tret.transpose((1, 0, 2))
                        tret = tret.reshape(list(sizXY) + list(tret.shape[1:]))
                        tmskSlc = (tret[:, :, :, 1] > 0.5).astype(np.float)
                        tsegm3D[:, :, sslst] = tmskSlc

                # self.dataImg[ii] = tdataImg
            # (4) inference
            lstIdx = range(numImg)
            splitIdx = split_list_by_blocks(lstIdx, batchSize)
            ret = []
            for ss in splitIdx:
                dataX = np.zeros([len(ss)] + list(self.shapeImg), dtype=np.float)
                for ii, ssi in enumerate(ss):
                    dataX[ii] = self.dataImg[ssi]
                retY = self.model.predict_on_batch(dataX)
                if self.isTheanoShape:
                    retY = retY.transpose((0, 2, 1))
                for ii in range(retY.shape[0]):
                    ret.append(retY[ii].reshape(self.shapeMsk))
            self.isDataInMemory = False
            self.dataImg = None
            return ret
        else:
            return []


######################################################
if __name__=='__main__':
    fidxTrn = '../../experimental_data/resize-256x256x64/idx.txt-train.txt'
    fidxVal = '../../experimental_data/resize-256x256x64/idx.txt-val.txt'
    #
    if K.image_dim_ordering() == 'tf':
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        set_session(tf.Session(config=config))
    #
    batcherTrain = BatcherOnImageCT3D(pathDataIdx=fidxTrn, isTheanoShape=False, numSlices=3, isLoadIntoMemory=True)
    print (batcherTrain)
    # dataX,dataY=batcherTrain.getSlice25D(0, [12,13,14,15,16])
    dictDataX, dictDataY = batcherTrain.getBatchDataSliced(isReturnDict=True)
    for kk in dictDataX.keys():
        tx = dictDataX[kk]
        ty = dictDataY[kk]
        print ('item-DataX.shape = %s, item-DataY.shape = %s' % (list(tx.shape), list(ty.shape)))
    dataX, dataY = batcherTrain.getBatchDataSliced(parNumImages=4, parNumSlices=8, isReturnDict=False)
    if K.image_dim_ordering()=='th':
        timg = dataX[0][dataX.shape[1]/2,:,:]
        tlbl = dataY[0][:,1].reshape(timg.shape)
    else:
        timg = dataX[0][:, :, dataX.shape[-1] / 2]
        tlbl = dataY[0][:, 1].reshape(timg.shape)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(timg), plt.title('CT-Image')
    plt.subplot(1, 2, 2)
    plt.imshow(tlbl), plt.title('CT-Mask')
    plt.show()
    print ('----')
    print ('DataX.shape = %s, DataY.shape = %s' % (list(dataX.shape), list(dataY.shape)))



