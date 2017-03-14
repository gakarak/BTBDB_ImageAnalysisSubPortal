#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar (Alexander Kalinovsky)'

import glob
import os
import sys
import time
import numpy as np
import json

import nibabel as nib

try:
   import cPickle as pickle
except:
   import pickle

import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skolor
import pandas as pd
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras.utils import np_utils

######################################################
def split_list_by_blocks(lst, psiz):
    tret = [lst[x:x + psiz] for x in range(0, len(lst), psiz)]
    return tret

######################################################
class BatcherCTLesion3D:
    pathDataIdx=None
    pathMeanData=None
    meanPrefix='mean.pkl'
    arrPathDataImg=None
    arrPathDataMsk=None
    wdir=None
    dataImg     = None
    dataMsk     = None
    dataMskCls  = None
    meanData = None
    #
    imgScale  = 1.
    modelPrefix = None
    #
    isTheanoShape=True
    isRemoveMeanImage=False
    isDataInMemory=False
    shapeImg = None
    numCh = 1
    numImg = -1
    # for inference
    model = None
    modelPath = None
    def __init__(self, pathDataIdx=None, pathMeanData=None, isRecalculateMeanIfExist=False,
                 isTheanoShape=True,
                 isRemoveMeanImage=False,
                 isLoadIntoMemory=False):
        if pathDataIdx is not None:
            self.loadDataset(pathDataIdx=pathDataIdx,
                             pathMeanData=pathMeanData,
                             isRecalculateMeanIfExist=isRecalculateMeanIfExist,
                             isTheanoShape=isTheanoShape,
                             isRemoveMeanImage=isRemoveMeanImage,
                             isLoadIntoMemory=isLoadIntoMemory)
    def loadDataset(self, pathDataIdx, pathMeanData=None, isRecalculateMeanIfExist=False,
                 isTheanoShape=True,
                 isRemoveMeanImage=False,
                 isLoadIntoMemory=False):
        self.isTheanoShape=isTheanoShape
        self.isRemoveMeanImage=isRemoveMeanImage
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
        tdataMsk = self.transformImageFromOriginal(tdataMsk>0, isRemoveMean=False)
        self.numCls = len(np.unique(tdataMsk))
        tdataMskCls = self.convertMskToOneHot(tdataMsk)
        self.shapeImg = tdataImg.shape
        self.shapeMsk = tdataMskCls.shape
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
            self.isDataInMemory = True
            self.dataImg = np.zeros([self.numImg] + list(self.shapeImg), dtype=np.float)
            self.dataMsk = np.zeros([self.numImg] + list(self.shapeImg), dtype=np.float)
            self.dataMskCls = np.zeros([self.numImg] + list(self.shapeMsk), dtype=np.float)
            print (':: Loading data into memory:')
            for ii in range(self.numImg):
                tpathImg = self.arrPathDataImg[ii]
                tpathMsk = self.arrPathDataMsk[ii]
                #
                tdataImg = self.adjustImage(nib.load(tpathImg).get_data())
                tdataMsk = nib.load(tpathMsk).get_data()
                tdataImg = self.transformImageFromOriginal(tdataImg, isRemoveMean=True)
                tdataMsk = self.transformImageFromOriginal(tdataMsk > 0, isRemoveMean=False)
                tdataMskCls = self.convertMskToOneHot(tdataMsk)
                self.dataImg[ii] = tdataImg
                self.dataMsk[ii] = tdataMsk
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
                tdataMsk = self.transformImageFromOriginal(tdataMsk > 0, isRemoveMean=False)
                tdataMskCls = self.convertMskToOneHot(tdataMsk)
                dataX[ii] = tdataImg
                dataY[ii] = tdataMskCls
        if self.isTheanoShape:
            tshp = dataY.shape
            dataY = dataY.reshape([tshp[0], tshp[1], np.prod(tshp[-3:])]).transpose((0, 2, 1))
            # print (tshp)
        else:
            tshp = dataY.shape
            dataY = dataY.reshape([tshp[0], np.prod(tshp[1:-1]), tshp[-1]])
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
    # def buildModel_TF(self, targetImageShaped=None):
    #     if not self.checkIsInitialized():
    #         retModel = buildModelOnImageCT_TF(inpShape=self.shapeImg, numCls=self.numCls)
    #         print ('>>> BatcherOnImage2D::buildModel() with input shape: %s' % list(retModel[0].input_shape) )
    #         return retModel
    #     else:
    #         raise Exception('*** BatcherOnImage2D is not initialized ***')
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
            model = keras.models.model_from_json(tmpStr)
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
        return BatcherCTLesion3D.loadModelFromJson(pathJson)
    def loadModelForInference(self, pathModelJson, pathMeanData, paramFilter=None):
        if os.path.isdir(pathModelJson):
            self.model = self.loadModelFromDir(pathModelJson)
        else:
            self.model = BatcherCTLesion3D.loadModelFromJson(pathModelJson)
        if os.path.isdir(pathMeanData):
            if paramFilter is None:
                lstMean = sorted(glob.glob('%s/*mean.pkl' % pathMeanData))
            else:
                lstMean = sorted(glob.glob('%s/*%s*mean.pkl' % (pathMeanData, paramFilter)))
            if len(lstMean)<1:
                raise Exception('Cant find mean-file in directory [%s]' % pathMeanData)
            self.pathMeanData = lstMean[0]
        else:
            self.pathMeanData = pathMeanData
        self.precalculateAndLoadMean(isRecalculateMean=False)
        self.numCls = self.model.output_shape[-1]
        self.shapeImg = self.model.input_shape[1:]
        self.isTheanoShape = (K.image_dim_ordering()=='th')
        if self.isTheanoShape:
            self.shapeMsk = tuple([self.numCls] + list(self.shapeImg[1:]))
        else:
            self.shapeMsk = tuple(list(self.shapeImg[:-1]) + [self.numCls])
    def inference(self, lstData, batchSize=2):
        if self.model is None:
            raise Exception('Model is not loaded... load model before call inferece()')
        if len(lstData)>0:
            tmpListOfImg = []
            # (1) load into memory
            if isinstance(lstData[0],str) or isinstance(lstData[0],unicode):
                for ii in lstData:
                    tmpListOfImg.append(nib.load(ii).get_data())
            elif isinstance(lstData[0], nib.nifti1.Nifti1Image):
                for ii in lstData:
                    tmpListOfImg.append(ii.get_data())
            else:
                tmpListOfImg = lstData
            # (2) check shapes
            tsetShapes = set()
            for ii in tmpListOfImg:
                tsetShapes.add(ii.shape)
            if len(tsetShapes)>1:
                raise Exception('Shapes of images must be equal sized')
            if K.image_dim_ordering()=='th':
                tmpShape = self.shapeImg[+1:]
            else:
                tmpShape = self.shapeImg[:-1]
            if tmpShape not in tsetShapes:
                raise Exception('Model input shape and shapes of input images is not equal!')
            # (3) convert data
            self.isDataInMemory = True
            numImg = len(tmpListOfImg)
            self.dataImg = np.zeros([numImg] + list(self.shapeImg), dtype=np.float)
            for ii in range(numImg):
                tdataImg = self.adjustImage(tmpListOfImg[ii])
                tdataImg = self.transformImageFromOriginal(tdataImg, isRemoveMean=True)
                self.dataImg[ii] = tdataImg
            # (4) inference
            lstIdx = range(numImg)
            splitIdx = split_list_by_blocks(lstIdx, batchSize)
            ret = []
            for ss in splitIdx:
                dataX = np.zeros([len(ss)] + list(self.shapeImg), dtype=np.float)
                for ii,ssi in enumerate(ss):
                    dataX[ii] = self.dataImg[ssi]
                retY = self.model.predict_on_batch(dataX)
                if self.isTheanoShape:
                    retY = retY.transpose((0,2,1))
                for ii in range(retY.shape[0]):
                    ret.append(retY[ii].reshape(self.shapeMsk))
            self.isDataInMemory = False
            self.dataImg = None
            return ret
        else:
            return []

#########################################
if __name__ == '__main__':
    pass