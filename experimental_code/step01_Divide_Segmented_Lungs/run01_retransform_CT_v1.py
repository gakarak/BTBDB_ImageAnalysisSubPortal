#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import nibabel as nib
import numpy as np

if __name__ == '__main__':
    wdir = '../../experimental_data/original'
    lstNii = sorted(glob.glob('%s/*[0-9].nii.gz' % wdir))
    numNii = len(lstNii)
    for ii,pp in enumerate(lstNii):
        pathImg = pp
        pathMsk = '%s-segmb.nii.gz' % pp
        timg = nib.load(pathImg)
        tmsk = nib.load(pathMsk)
        tout = nib.Nifti1Image(tmsk.get_data().astype(np.int16), timg.affine, header=timg.header)
        pathOut = '%s-segmbT.nii.gz' % pp
        nib.save(tout, pathOut)
        print ('[%d/%d] : converting %s -> %s' % (ii, numNii, os.path.basename(pathMsk), os.path.basename(pathOut)))
    print ('---')