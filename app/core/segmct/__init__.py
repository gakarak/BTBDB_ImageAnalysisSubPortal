#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import nibabel as nib

from fcnn_lung2d import BatcherCTLung2D
from fcnn_lesion3d import BatcherCTLesion3D

from app.core.preprocessing import resizeNii, resize3D

#########################################
def segmentLungs25D(pathInpNii, dirWithModel, pathOutNii=None, outSize=None, batchSize=8, isDebug=False, threshold=None):
    if isinstance(pathInpNii,str) or isinstance(pathInpNii,unicode):
        isInpFromFile = True
        if not os.path.isfile(pathInpNii):
            raise Exception('Cant find input file [%s]' % pathInpNii)
    else:
        isInpFromFile = False
    if not os.path.isdir(dirWithModel):
        raise Exception('Cant find directory with model [%s]' % dirWithModel)
    if pathOutNii is not None:
        outDir = os.path.dirname(os.path.abspath(pathOutNii))
        if not os.path.isdir(outDir):
            raise Exception('Cant find output directory [%s], create directory for output file before this call' % outDir)
    batcherInfer = BatcherCTLung2D()
    batcherInfer.loadModelForInference(pathModelJson=dirWithModel, pathMeanData=dirWithModel)
    if isDebug:
        batcherInfer.model.summary()
    lstPathNifti = [ pathInpNii ]
    ret = batcherInfer.inference(lstPathNifti, batchSize=batchSize, isDebug=isDebug)
    outMsk = ret[0]
    if isInpFromFile:
        tmpNii = nib.load(pathInpNii)
    else:
        tmpNii = pathInpNii
    #
    outMskNii = nib.Nifti1Image(outMsk.copy().astype(np.float16), tmpNii.affine, header=tmpNii.header)
    # resize if need:
    if outSize is not None:
        outMskNii = resizeNii(outMskNii, newSize=outSize)
    # threshold if need:
    if threshold is not None:
        outMskNii = nib.Nifti1Image( (outMskNii.get_data()>threshold).astype(np.float16), outMskNii.affine, header=outMskNii.header)
    # save if output path is present
    if pathOutNii is not None:
        nib.save(outMskNii, pathOutNii)
        # pathOutNii = '%s-segm.nii.gz' % pathInpNii
    else:
        return outMskNii

#########################################
def segmentLesions3D(pathInpNii, dirWithModel, pathOutNii=None, outSize=None, isDebug=False, threshold=None):
    if isinstance(pathInpNii, str) or isinstance(pathInpNii, unicode):
        isInpFromFile = True
        if not os.path.isfile(pathInpNii):
            raise Exception('Cant find input file [%s]' % pathInpNii)
    else:
        isInpFromFile = False
    if not os.path.isdir(dirWithModel):
        raise Exception('Cant find directory with model [%s]' % dirWithModel)
    if pathOutNii is not None:
        outDir = os.path.dirname(os.path.abspath(pathOutNii))
        if not os.path.isdir(outDir):
            raise Exception(
                'Cant find output directory [%s], create directory for output file before this call' % outDir)
    batcherInfer = BatcherCTLesion3D()
    batcherInfer.loadModelForInference(pathModelJson=dirWithModel, pathMeanData=dirWithModel)
    if isDebug:
        batcherInfer.model.summary()
    ret = batcherInfer.inference([pathInpNii], batchSize=1)
    if batcherInfer.isTheanoShape:
        outMsk = ret[0][1, :, :, :]
    else:
        outMsk = ret[0][:, :, :, 1]
    if isInpFromFile:
        tmpNii = nib.load(pathInpNii)
    else:
        tmpNii = pathInpNii
    #
    outMskNii = nib.Nifti1Image(outMsk.copy().astype(np.float16), tmpNii.affine, header=tmpNii.header)
    if outSize is not None:
        outMskNii = resizeNii(outMskNii, newSize=outSize)
    if threshold is not None:
        outMskNii = nib.Nifti1Image((outMskNii.get_data() > threshold).astype(np.float16),
                                    outMskNii.affine,
                                    header=outMskNii.header)
    if pathOutNii is not None:
        nib.save(outMskNii, pathOutNii)
        # pathOutNii = '%s-segm.nii.gz' % pathInpNii
    else:
        return outMskNii

#########################################
def api_segmentLungAndLesion(dirModelLung, dirModelLesion, series,
                             ptrLogger=None,
                             shape4Lung = (256, 256, 64), shape4Lesi = (128, 128, 64)):
    # (1) msg-helpers
    def msgInfo(msg):
        if ptrLogger is not None:
            ptrLogger.info(msg)
        else:
            print (msg)
    def msgErr(msg):
        if ptrLogger is not None:
            ptrLogger.error(msg)
        else:
            print (msg)
    # (2.1) check data
    if not series.isInitialized():
        msgErr('Series is not initialized, skip .. [{0}]'.format(series))
        return False
    # if not series.isDownloaded():
    #     msgErr('Series data is not downloaded, skip .. [{0}]'.format(series))
    #     return False
    if not series.isConverted():
        msgErr('Series DICOM data is not converted to Nifti format, skip .. [{0}]'.format(series))
        return False
    # (2.2) check existing files
    pathNii = series.pathConvertedNifti(isRelative=False)
    pathSegmLungs = series.pathPostprocLungs(isRelative=False)
    pathSegmLesions = series.pathPostprocLesions(isRelative=False)
    if os.path.isfile(pathSegmLungs) and os.path.isfile(pathSegmLesions):
        msgInfo('Series data is already segmented, skip task ... [{0}]'.format(series))
        return False
    else:
        # (2.3.1) load and resize
        try:
            dataNii = nib.load(pathNii)
            shapeOrig = dataNii.shape
            niiResiz4Lung = resizeNii(dataNii, shape4Lung)
            niiResiz4Lesi = resizeNii(dataNii, shape4Lesi)
        except Exception as err:
            msgErr('Cant load and resize input nifti file [{0}] : {1}, for series [{2}]'.format(pathNii, err, series))
            return False
        # (2.3.2) segment lungs
        try:
            lungMask = segmentLungs25D(niiResiz4Lung,
                                       dirWithModel=dirModelLung,
                                       pathOutNii=None,
                                       outSize=shapeOrig,
                                       # outSize=shape4Lung,
                                       threshold=0.5)
        except Exception as err:
            msgErr('Cant segment lungs for file [{0}] : {1}, for series [{2}]'.format(pathNii, err, series))
            return False
        # (2.3.3) segment lesions
        try:
            lesionMask = segmentLesions3D(niiResiz4Lesi,
                                          dirWithModel=dirModelLesion,
                                          pathOutNii=None,
                                          outSize=shapeOrig,
                                          # outSize=shape4Lung,
                                          threshold=None)
        except Exception as err:
            msgErr('Cant segment lesions for file [{0}] : {1}, for series [{2}]'.format(pathNii, err, series))
            return False
        # (2.3.4) save results
        try:
            nib.save(lungMask, pathSegmLungs)
            nib.save(lesionMask, pathSegmLesions)
        except Exception as err:
            msgErr('Cant save segmentation results to file [{0}] : {1}, for series [{2}]'.format(pathSegmLesions, err, series))
            return False
        return True

#########################################
if __name__ == '__main__':
    print ('---')