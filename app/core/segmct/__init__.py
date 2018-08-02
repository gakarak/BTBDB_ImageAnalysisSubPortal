#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import nibabel as nib

from app.core.segmct.fcnn_lung2d import BatcherCTLung2D
from app.core.segmct.fcnn_lesion3d import BatcherCTLesion3D
from app.core.segmct.fcnn_lesion3dv2 import Inferencer as InferencerLesion3Dv2, lesion_id2name, lesion_id2rgb, lesion_name2id

import json
import skimage.io as skio
import app.core.preprocessing as preproc
from app.core.preprocessing import resizeNii, resize3D

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import app.core.lesion_descriptors as ldsc

#########################################
def segmentLungs25D(pathInpNii, dirWithModel, pathOutNii=None, outSize=None, batchSize=8, isDebug=False, threshold=None):
    if isinstance(pathInpNii,str):# or isinstance(pathInpNii,unicode):
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
    if isinstance(pathInpNii, str):# or isinstance(pathInpNii, unicode):
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
def segmentLesions3Dv2(pathInpNii, dirWithModel, pathOutNii=None, outSize=None, isDebug=False, threshold=None, path_lungs=None):
    if isinstance(pathInpNii, str):# or isinstance(pathInpNii, unicode):
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
    batcherInfer = InferencerLesion3Dv2()
    batcherInfer.load_model(path_model=dirWithModel)
    if isDebug:
        batcherInfer.model.summary()
    ret = batcherInfer.inference([pathInpNii], batchSize=1)
    outMsk = ret[0]
    if isInpFromFile:
        tmpNii = nib.load(pathInpNii)
    else:
        tmpNii = pathInpNii
    #
    outMskNii = nib.Nifti1Image(outMsk.copy().astype(np.uint8), tmpNii.affine, header=tmpNii.header)
    if outSize is not None:
        outMskNii = resizeNii(outMskNii, newSize=outSize, parOrder = 0)
    if path_lungs is not None:
        tmp_affine = outMskNii.affine
        tmp_header = outMskNii.header
        msk_lungs = resizeNii(path_lungs, newSize=outSize, parOrder=0).get_data()
        outMsk = outMskNii.get_data().astype(np.uint8)
        outMsk[msk_lungs < 0.5] = 0
        outMskNii = nib.Nifti1Image(outMsk.copy().astype(np.uint8), tmp_affine, header=tmp_header)
    # if threshold is not None:
    #     outMskNii = nib.Nifti1Image((outMskNii.get_data() > threshold).astype(np.float16),
    #                                 outMskNii.affine,
    #                                 header=outMskNii.header)
    if pathOutNii is not None:
        nib.save(outMskNii, pathOutNii)
        # pathOutNii = '%s-segm.nii.gz' % pathInpNii
    else:
        return outMskNii

#########################################
def api_segmentLungAndLesion(dirModelLung, dirModelLesion, series,
                             ptrLogger=None,
                             shape4Lung = (256, 256, 64), shape4Lesi = (256, 256, 64), gpuMemUsage=0.4):
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
    # pathSegmLungs = series.pathPostprocLungs(isRelative=False)
    pathSegmLungs = series.pathPostprocLungs(isRelative=False)
    pathSegmLesions = series.pathPostprocLesions2(isRelative=False)
    if os.path.isfile(pathSegmLungs) and os.path.isfile(pathSegmLesions):
        msgInfo('Series data is already segmented, skip task ... [{0}]'.format(series))
        return False
    else:
        # (2.3.0) TF GPU memory usage constraints
        # FIXME:temporary fix, in future: apped memory usage parameter in application config
        import tensorflow as tf
        import keras.backend as K
        from keras.backend.tensorflow_backend import set_session
        if K.image_dim_ordering() == 'tf':
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = gpuMemUsage
            set_session(tf.Session(config=config))

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
            if not os.path.isfile(pathSegmLungs):
                lungMask = segmentLungs25D(niiResiz4Lung,
                                           dirWithModel=dirModelLung,
                                           pathOutNii=None,
                                           outSize=shapeOrig,
                                           # outSize=shape4Lung,
                                           threshold=0.5)
                nib.save(lungMask, pathSegmLungs)
            else:
                pass
                # lungMask = nib.load(pathSegmLungs)
        except Exception as err:
            msgErr('Cant segment lungs for file [{0}] : {1}, for series [{2}]'.format(pathNii, err, series))
            return False
        # (2.3.3) segment lesions
        try:
            # lesionMask = segmentLesions3D(niiResiz4Lesi,
            #                               dirWithModel=dirModelLesion,
            #                               pathOutNii=None,
            #                               outSize=shapeOrig,
            #                               # outSize=shape4Lung,
            #                               threshold=None)
            lesionMask = segmentLesions3Dv2(niiResiz4Lesi,
                                            dirWithModel=dirModelLesion,
                                            pathOutNii=None,
                                            outSize=shapeOrig,
                                            # outSize=shape4Lung,
                                            threshold=None,
                                            path_lungs=pathSegmLungs)
        except Exception as err:
            msgErr('Cant segment lesions for file [{0}] : {1}, for series [{2}]'.format(pathNii, err, series))
            return False
        # (2.3.4) save results
        try:
            nib.save(lesionMask, pathSegmLesions)
        except Exception as err:
            msgErr('Cant save segmentation results to file [{0}] : {1}, for series [{2}]'.format(pathSegmLesions, err, series))
            return False
        return True

def api_generateAllReports(series,
                           dirModelLung, dirModelLesion,
                           ptrLogger=None,
                           shape4Lung = (256, 256, 64), shape4Lesi = (256, 256, 64)):
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
    # (0) prepare path-variables
    pathNii = series.pathConvertedNifti(isRelative=False)
    pathSegmLungs = series.pathPostprocLungs(isRelative=False)
    pathSegmLungsDiv2 = series.pathPostprocLungsDiv2(isRelative=False)
    # pathSegmLesions1 = series.pathPostprocLesions(isRelative=False)
    pathSegmLesions1 = series.pathPostprocLesions2(isRelative=False)
    pathPreview2 = series.pathPostprocPreview(isRelative=False, previewId=2)
    pathPreview3 = series.pathPostprocPreview(isRelative=False, previewId=3)
    pathPreview4 = series.pathPostprocPreview(isRelative=False, previewId=4)
    pathReport = series.pathPostprocReport(isRelative=False)
    # (1) Lung/Lesions segmentation
    retSegm = api_segmentLungAndLesion(dirModelLung=dirModelLung,
                             dirModelLesion=dirModelLesion,
                             series=series,
                             ptrLogger=ptrLogger,
                             shape4Lung=shape4Lung,
                             shape4Lesi=shape4Lesi)
    msgInfo('Segmentation Lung/Lesion isOk = {0}'.format(retSegm))
    if (not os.path.isfile(pathSegmLungs)) or (not os.path.isfile(pathSegmLesions1)):
        msgErr('Cant segment Lung/Lesion, skip... [{0}]'.format(series))
        return False
    # (1.1) loading lungs-masks/lesions-masks
    try:
        niiLung = nib.load(pathSegmLungs)
        niiLesion = nib.load(pathSegmLesions1)
    except Exception as err:
        msgErr('Cant load Lung/Lesion Nifti data: [{0}], for {1}'.format(err, pathSegmLesions1))
        return False
    # (2) prepare divided lungs
    if not os.path.isfile(pathSegmLungsDiv2):
        niiLungDiv, _ = preproc.makeLungedMaskNii(niiLung)
        nib.save(niiLungDiv, pathSegmLungsDiv2)
    else:
        niiLungDiv = nib.load(pathSegmLungsDiv2)
    # (3) calc lesion score
    try:
        retLesionScoreBin, retLesionScoreById, retLesionScoreByName = preproc.prepareLesionDistribInfoV2(niiLung, niiLesion, niiLungDIV2=niiLungDiv)
    except Exception as err:
        msgErr('Cant evaluate Lesion-score: [{0}], for {1}'.format(err, pathSegmLesions1))
        return False
    # (3.1) calc cbir-descriptor
    try:
        cbir_desc = ldsc.calc_desc(pathSegmLungsDiv2, pathSegmLesions1)
        cbir_desc_json = ldsc.desc_to_json(cbir_desc)
    except Exception as err:
        msgErr('Cant evaluate Lesion-score: [{0}], for {1}'.format(err, pathSegmLesions1))
        return False
    # (4) prepare short report about lungs
    try:
        retLungInfo = preproc.prepareLungSizeInfoNii(niiLungDiv)
    except Exception as err:
        msgErr('Cant get Lung information : [{0}], for {1}'.format(err, series))
        return False
    # (5) generate preview & save preview image
    # try:
    dataImg = preproc.normalizeCTImage(nib.load(pathNii).get_data())
    dataMsk = niiLung.get_data()
    dataLes = niiLesion.get_data()

    imgPreviewJson2 = preproc.genPreview2D(dataImg, dataMsk, dataLes, pathPreview2, 2)
    imgPreviewJson3 = preproc.genPreview2D(dataImg, dataMsk, dataLes, pathPreview3, 3)
    # imgPreviewJson4 = preproc.genPreview2D(dataImg, dataMsk, dataLes, pathPreview4, 4)

    # (6) generate & save JSON report
    try:
        jsonReport = preproc.getJsonReport(series=series,
                                           reportLesionScore=None, #retLesionScoreBin,
                                           reportLesion=cbir_desc_json,
                                           reportLungs=retLungInfo,
                                           lstImgJson=[imgPreviewJson3, imgPreviewJson2],
                                           reportLesionScoreById = None, #retLesionScoreById,
                                           reportLesionScoreByName = None) #retLesionScoreByName)
        with open(pathReport, 'w') as f:
            f.write(json.dumps(jsonReport, indent=4))
    except Exception as err:
        msgErr('Cant generate final JSON report : [{0}], for {1}'.format(err, series))
        return False

    # (7) generate and save 3 directories with DICOM files, converted from Lesions NifTi
    #     original, lesions_only and lesions_map
    #     file names convention: {S3 bucket name}/viewer/{map_type}/{patientID}/{studyUID}/{seriesUID}/{instanceUID}.{extension}
    # preproc.prepareCTpreview(series)

    # FIXME: append PDF generation in future here
    # (6) generate PDF preview
    return True

def api_generateCBIR_BuildDSC(db_watcher, num_threads):


    print('-')


#########################################
if __name__ == '__main__':
    print ('---')