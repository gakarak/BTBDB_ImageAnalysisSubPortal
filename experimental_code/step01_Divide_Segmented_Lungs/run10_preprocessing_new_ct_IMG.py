#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import nibabel as nib
import numpy as np

if __name__ == '__main__':
    wdirInp = '../../experimental_data/data_analyze/original'
    wdirOut = '%s-nii' % wdirInp
    if not os.path.isdir(wdirOut):
        os.mkdir(wdirOut)
    lstAnalyze = glob.glob('%s/*.hdr' % wdirInp)
    numAnalyze = len(lstAnalyze)
    for ii,pp in enumerate(lstAnalyze):
        fout = '%s/%s.nii.gz' % (wdirOut, os.path.splitext(os.path.basename(pp))[0])
        print ('[%d/%d] : %s --> %s' % (ii, numAnalyze, os.path.basename(pp), os.path.basename(fout)))
        tdataInp  = nib.load(pp)
        newData   = tdataInp.get_data() + 0*tdataInp.header['scl_inter']
        taffine = tdataInp.affine
        newHeader = nib.Nifti1Header.from_header(tdataInp.header)
        newHeader['scl_inter'] = float('nan')
        newHeader.set_xyzt_units(xyz='mm')
        newHeader.set_sform(taffine)
        tdataOut  = nib.Nifti1Image(newData.astype(np.int16), tdataInp.affine, header=newHeader)
        nib.save(tdataOut, fout)
