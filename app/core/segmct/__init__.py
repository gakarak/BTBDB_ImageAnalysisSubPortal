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
def segmentLungs3D(pathInpNii, dirWithModel, pathOutNii=None, outSize=None, batchSize=8, isDebug=False, threshold=None):
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
if __name__ == '__main__':
    print ('---')