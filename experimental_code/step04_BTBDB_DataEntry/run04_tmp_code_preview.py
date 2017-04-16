#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import nibabel as nib
import app.core.preprocessing as preproc

if __name__ == '__main__':
    fnii = '../../experimental_data/dataentry_test0/case-2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0/study-dd10657e-f2c3-48ba-87d6-b5f3fc40c752/series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2-CT.nii.gz'
    fniiMsk = '%s-lungs.nii.gz' % fnii
    fniiLes = '%s-lesion.nii.gz' % fnii
    sizPrv = 256
    shpPrv = (sizPrv, sizPrv, sizPrv)
    dataImg = preproc.niiImagePreTransform(nib.load(fnii).get_data())
    dataMsk = preproc.niiImagePreTransform(nib.load(fniiMsk).get_data())
    dataLes = preproc.niiImagePreTransform(nib.load(fniiLes).get_data())
    #
    dataImgR = preproc.resize3D(dataImg, shpPrv)
    dataMskR = preproc.resize3D(dataMsk, shpPrv)
    dataLesR = preproc.resize3D(dataLes, shpPrv)

    print('--')
