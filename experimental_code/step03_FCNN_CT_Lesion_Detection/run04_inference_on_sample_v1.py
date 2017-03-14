#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import matplotlib.pyplot as plt
import numpy as np

from run00_common import BatcherOnImageCT3D

#########################################
def createLesionMask(pathInpNii, dirWithModel, pathOutNii=None, isDebug=False):
    if not os.path.isfile(pathInpNii):
        raise Exception('Cant find input file [%s]' % pathInpNii)
    if not os.path.isdir(dirWithModel):
        raise Exception('Cant find directory with model [%s]' % dirWithModel)
    if pathOutNii is not None:
        outDir = os.path.dirname(os.path.abspath(pathOutNii))
        if not os.path.isdir(outDir):
            raise Exception(
                'Cant find output directory [%s], create directory for output file before this call' % outDir)
    #
    batcherInfer = BatcherOnImageCT3D()
    batcherInfer.loadModelForInference(pathModelJson=dirWithModel, pathMeanData=dirWithModel)
    if isDebug:
        batcherInfer.model.summary()
    ret = batcherInfer.inference([pathInpNii], batchSize=1)
    print ('----')


#########################################
if __name__ == '__main__':
    pathDirWithModels = '../../experimental_data/models/fcnn_ct_lesion_segm_3d'
    lstPathNifti = [
        '../../experimental_data/TB_sub_1_5-resize-128x128x64_3case/data/tb1_001_1001_1_33554433-128x128x64.nii.gz',
        '../../experimental_data/TB_sub_1_5-resize-128x128x64_3case/data/tb1_002_10064_1_33554433-128x128x64.nii.gz',
        '../../experimental_data/TB_sub_1_5-resize-128x128x64_3case/data/tb1_003_102_2_16777217-128x128x64.nii.gz',
    ]
    for ii, pp in enumerate(lstPathNifti):
        print ('[%d/%d] : %s' % (ii, len(lstPathNifti), pp))
        foutNii = '%s-msk.nii.gz' % (os.path.basename(pp))
        createLesionMask(pathInpNii=pp, dirWithModel=pathDirWithModels, pathOutNii=foutNii, isDebug=True)