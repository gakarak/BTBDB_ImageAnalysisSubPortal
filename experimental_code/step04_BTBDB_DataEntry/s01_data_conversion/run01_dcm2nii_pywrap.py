#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

from app.core.utils.cmd import pydcm2nii

#####################################
if __name__ == '__main__':
    dirWithDcm = '../../../experimental_data/dataentry_test0/case-2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0/study-dd10657e-f2c3-48ba-87d6-b5f3fc40c752/series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2/raw'
    foutNii = 'test-out.nii.gz'
    # pexe='dcm2nii'
    print ('pydcm2nii isOk = %s' % pydcm2nii(dirWithDcm, foutNii))
