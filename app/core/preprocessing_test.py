#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import nibabel as nib
import app.core.preprocessing as preproc

import unittest

class TestLungDividing(unittest.TestCase):
    def setUp(self):
        self.wdir = '../../experimental_data/resize-256x256x64'

    def test_resize_nii(self):
        self.assertTrue(os.path.isdir(self.wdir))
        lstPathNii = sorted(glob.glob('%s/*.nii.gz' % self.wdir))
        numNii = len(lstPathNii)
        self.assertTrue((numNii > 0))
        pathNii  = lstPathNii[0]
        newSize = (128,128,128)
        niiResiz = preproc.resizeNii(pathNii, newSize=newSize)
        self.assertTrue(niiResiz.shape == newSize)

    def test_divide_morphological(self):
        self.assertTrue(os.path.isdir(self.wdir))
        lstPathNii = sorted(glob.glob('%s/*-msk.nii.gz' % self.wdir))
        numNii = len(lstPathNii)
        self.assertTrue( (numNii>0) )
        for ii,pathNii in enumerate(lstPathNii):
            timg = preproc.niiImagePreTransform(nib.load(pathNii).get_data())
            retMskLungs, retIsOk = preproc.makeLungedMask(timg)
            # (1) check, that #lungs is non-zero for test-images
            self.assertTrue(len(np.unique(retMskLungs))>0)
            # (2) check ret-result: retIsOk=False if only one lung in data
            tmp = np.unique(retMskLungs)
            numLungs = int(np.sum( (tmp>0)&(tmp<3)))
            if retIsOk:
                self.assertTrue(numLungs, 2)
            else:
                self.assertTrue( numLungs, 1)
            print ('\t[%d/%d] %s, #Lungs = %d, isOk = %s' % (ii, numNii, os.path.basename(pathNii), numLungs, retIsOk))

    def test_lung_lesion_report(self):
        tmpDir = '../../experimental_data/dataentry_test0/case-2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0/study-dd10657e-f2c3-48ba-87d6-b5f3fc40c752'
        fmskLung = os.path.join(tmpDir, 'series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2-CT.nii.gz-lungs.nii.gz')
        fmskLesion = os.path.join(tmpDir, 'series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2-CT.nii.gz-lesion.nii.gz')
        #
        niiLung = nib.load(fmskLung)
        niiLesion = nib.load(fmskLesion)
        retInfo = preproc.prepareLesionDistribInfo(niiLung, niiLesion)
        self.assertTrue(len(retInfo)>1)

    def test_preview_generation(self):
        tmpDir = '../../experimental_data/dataentry_test0/case-2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0/study-dd10657e-f2c3-48ba-87d6-b5f3fc40c752'
        fimgLung = '%s/series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2-CT.nii.gz' % tmpDir
        fmskLung = '%s/series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2-CT.nii.gz-lungs.nii.gz' % tmpDir
        fmskLesion = '%s/series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2-CT.nii.gz-lesion.nii.gz' % tmpDir
        #
        dataImg = preproc.normalizeCTImage(nib.load(fimgLung).get_data())
        dataMsk = nib.load(fmskLung).get_data()
        dataLes = nib.load(fmskLesion).get_data()
        #
        imgPreview = preproc.makePreview4Lesion(dataImg, dataMsk, dataLes)
        self.assertTrue(np.min(imgPreview.shape[:2])>256)

if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLungDividing)
    unittest.TextTestRunner(verbosity=2).run(suite)
