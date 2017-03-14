#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar (Alexander Kalinovsky)'

import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from run00_common import BatcherOnImageCT3D, split_list_by_blocks

#########################################
def createLungMask(pathInpNii, dirWithModel, pathOutNii=None, batchSize=8, isDebug=False):
    if not os.path.isfile(pathInpNii):
        raise Exception('Cant find input file [%s]' % pathInpNii)
    if not os.path.isdir(dirWithModel):
        raise Exception('Cant find directory with model [%s]' % dirWithModel)
    if pathOutNii is not None:
        outDir = os.path.dirname(os.path.abspath(pathOutNii))
        if not os.path.isdir(outDir):
            raise Exception('Cant find output directory [%s], create directory for output file before this call' % outDir)
    batcherInfer = BatcherOnImageCT3D()
    batcherInfer.loadModelForInference(pathModelJson=dirWithModel, pathMeanData=dirWithModel)
    if isDebug:
        batcherInfer.model.summary()
    lstPathNifti = [ pathInpNii ]
    ret = batcherInfer.inference(lstPathNifti, batchSize=batchSize, isDebug=isDebug)
    outMsk = ret[0]
    if pathOutNii is None:
        pathOutNii = '%s-segm.nii.gz' % pathInpNii
    tmpNii = nib.load(pathInpNii)
    outMskNii = nib.Nifti1Image(outMsk.copy().astype(np.float16), tmpNii.affine, header=tmpNii.header)
    nib.save(outMskNii, pathOutNii)

#########################################
if __name__ == '__main__':
    pathDirWithModels = '../../experimental_data/models/fcnn_ct_lung_segm_2.5d'
    lstPathNifti = [
        '../../experimental_data/resize-256x256x64/0009-256x256x64.nii.gz',
        '../../experimental_data/resize-256x256x64/0026-256x256x64.nii.gz',
        '../../experimental_data/resize-256x256x64/0177-256x256x64.nii.gz',
    ]
    for ii,pp in enumerate(lstPathNifti):
        print ('[%d/%d] : %s' % (ii,len(lstPathNifti), pp))
        foutNii = '%s-msk.nii.gz' % (os.path.basename(pp))
        createLungMask(pathInpNii=pp, dirWithModel=pathDirWithModels, pathOutNii=foutNii)
