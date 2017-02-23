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

if __name__ == '__main__':
    unittest.main()