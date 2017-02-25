#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar (Alexander Kalinovsky)'

import os

import nibabel as nib
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from run00_common import BatcherOnImageCT3D, split_list_by_blocks

def getMinMaxZinCTMask(imgCT, parThresh=0.01, pbrd=0.1):
    if imgCT.ndim<3:
        raise Exception('Invalid CT image: CT image must have 3 dim!')
    sizZ = imgCT.shape[-1]
    parMinZ = 5
    if sizZ<parMinZ:
        raise Exception('Invalid CT image: number of CT-slices must be at lest #%d' % parMinZ)
    arrSumZ  = np.sum(imgCT.astype(np.float),axis=(0,1)).reshape(-1)
    tsumZ    = np.sum(arrSumZ)
    tvolume  = np.prod(imgCT.shape)
    if tsumZ>(0.05*tvolume):
        arrSumZP    = np.cumsum(arrSumZ)/tsumZ
        zmin = np.nonzero(arrSumZP>parThresh)[0][0]
        zmax = np.nonzero(arrSumZP>(1.-parThresh))[0][0]
    else:
        zmin = int(sizZ * pbrd)
        zmax = int(sizZ * (1.-pbrd))
    return (zmin, zmax)

def getCTSlicesZ(imgCT, numSlices=5, parThresh=0.01):
    zmin, zmax = getMinMaxZinCTMask(imgCT, parThresh=parThresh)
    tret = np.floor(np.linspace(zmin, zmax, num=numSlices)).astype(np.int)
    return tret

def generatePanoImage(lstImages, numr=3, numc=3, isRepeat=False, brd=3):
    # (1) check equality of shapes of images
    setOfShapes = {ii.shape for ii in lstImages}
    if len(setOfShapes)>1:
        raise Exception('Invalid list of images: all shapes must be equal!')
    numTot = numr*numc
    numImg = len(lstImages)
    # (2) append pad-borders for images
    if brd<1:
        inpListfImages = lstImages
    else:
        inpListfImages = []
        for img in lstImages:
            if img.ndim <3:
                timgPad  = np.pad(img, brd, 'constant')
            else:
                timgPad  = np.pad(img, ((brd,brd), (brd,brd), (0,0)), 'constant')
            inpListfImages.append(timgPad)
    # (3) preprocess list of images if number of images is not equal to number of cells [numr x numc]
    if numTot<numImg:
        tlstImages = inpListfImages[:numTot]
    else:
        tlstImages = [ii for ii in inpListfImages]
        if isRepeat:
            cnt = 0
            for ii in range(numTot-numImg):
                tlstImages.append(inpListfImages[cnt])
                cnt += 1
                if cnt>=numImg:
                    cnt = 0
        else:
            timgZero = np.zeros(lstImages[0].shape, lstImages[0].dtype)
            if brd>0:
                if lstImages[0].ndim < 3:
                    timgZero = np.pad(timgZero, brd, 'constant')
                else:
                    timgZero = np.pad(timgZero, ((brd, brd), (brd, brd), (0, 0)), 'constant')
            for ii in range(numTot - numImg):
                tlstImages.append(timgZero)
    # (4) generate preview pattern
    ret = None
    cnt = 0
    for rr in range(numr):
        tmpMerge = []
        for cc in range(numc):
            tmpMerge.append(tlstImages[cnt])
            cnt +=1
        tmpMerge = np.hstack(tmpMerge)
        if ret is None:
            ret = tmpMerge
        else:
            ret = np.vstack([ret, tmpMerge])
    return ret

import matplotlib.pyplot as plt
import skimage.transform as sktf

#############################################################
if __name__ == '__main__':
    #
    if K.image_dim_ordering() == 'tf':
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.86
        set_session(tf.Session(config=config))
    #
    parNumSlices   = 2
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
    fidxTrn = '/mnt/data1T/datasets/CRDF/CT_with_segm_mask_v3/resize-256x256x64/idx.txt'
    fidxVal = '/mnt/data1T/datasets/CRDF/CT_with_segm_mask_v3/resize-256x256x64/idx.txt'
    # fidxTrn = '../../experimental_data/resize-256x256x64/idx.txt'
    # fidxVal = '../../experimental_data/resize-256x256x64/idx.txt'
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
        tmskImg  = tmsk.get_data().astype(np.float)/255.
        tsgmImg = tsegm.get_data().astype(np.float)
        tarrZ = getCTSlicesZ(tsegm.get_data(), 8)
        #
        tsiz = tmskImg.shape
        frontMsk = tmskImg[:, int(tsiz[1] / 2), :]
        frontSgm = tsgmImg[:, int(tsiz[1] / 2), :]
        frontMsk = sktf.resize(frontMsk, (tsiz[0], tsiz[0]))
        frontSgm = sktf.resize(frontSgm, (tsiz[0], tsiz[0]))
        frontImg = np.dstack((frontSgm,frontMsk,frontSgm))
        frontImg = np.rot90(frontImg)
        #
        tlstImages = [frontImg]
        for zz in tarrZ:
            imA = tsgmImg[:, :, zz]
            imB = tmskImg[:, :, zz]
            tlstImages.append(np.dstack((imA, imB, imA)))
        timgPano = generatePanoImage(tlstImages, 3, 3, isRepeat=True)
        plt.imshow(timgPano.astype(np.float))
        plt.axis('off')
        # plt.show()
        foutMsk = '%s-segm-%s-slc%d.nii.gz' % (tpathMsk, parModelType, parNumSlices)
        foutPrv = '%s-segm-%s-slc%d-preview.png' % (tpathMsk, parModelType, parNumSlices)
        plt.savefig(foutPrv, bbox_inches='tight')
        nib.save(tsegm, foutMsk)
        print ('\t[%d/%d] * processing : %s --> %s' % (ii, batcherVal.numImg, os.path.basename(tpathMsk), os.path.basename(foutMsk)))
        # tmsk = nib.load(tpathMsk)
