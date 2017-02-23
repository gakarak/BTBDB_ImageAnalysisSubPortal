#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import shutil
import os
import glob
import nibabel as nib
import numpy as np

if __name__ == '__main__':
    wdirIMG = '../../experimental_data/data_analyze/original-nii'
    wdirMSK = '../../experimental_data/data_analyze/resegm2'
    wdirOut = '%s-masked' % wdirIMG
    if not os.path.isdir(wdirOut):
        os.mkdir(wdirOut)
    lstAnalyze = sorted(glob.glob('%s/*.hdr' % wdirMSK))
    numAnalyze = len(lstAnalyze)
    for ii,pp in enumerate(lstAnalyze):
        tidx = os.path.basename(pp).split('_')[0]
        finpIMG = '%s/%s.nii.gz' % (wdirIMG, tidx)
        finpMSK = pp
        foutMSK = '%s/%s_resegm2.nii.gz' % (wdirOut, tidx)
        foutIMG = '%s/%s.nii.gz' % (wdirOut, tidx)
        #
        print ('[%d/%d] : %s --> %s' % (ii, numAnalyze, os.path.basename(pp), os.path.basename(foutMSK)))
        tniiIMG = nib.load(finpIMG)
        tniiMSK = nib.load(finpMSK)
        tdataOut = nib.Nifti1Image((255*(tniiMSK.get_data()>-3000)).astype(np.int16), tniiIMG.affine, header=tniiIMG.header)
        nib.save(tdataOut, foutMSK)
        shutil.copy(finpIMG, foutIMG)
