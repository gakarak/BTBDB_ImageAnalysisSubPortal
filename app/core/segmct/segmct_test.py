#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import unittest

import matplotlib.pyplot as plt
import nibabel as nib
from app.core.preprocessing import resizeNii
from fcnn_lung2d import BatcherCTLung2D
from app.core.segmct import segmentLesions3D, segmentLungs25D

class TestCTProcessing(unittest.TestCase):
    def setUp(self):
        self.wdir       = '../../../experimental_data'
        self.pathInpNii = '../../../experimental_data/original/tb5/tb1_001_1001_1_33554433.nii.gz'
        self.pathModelLung = '../../../experimental_data/models/fcnn_ct_lung_segm_2.5d_tf/'
        self.pathModelLesion = '../../../experimental_data/models/fcnn_ct_lesion_segm_3d_tf/'

    def test_lung_segmentation(self):
        #
        niiOriginal = nib.load(self.pathInpNii)
        inpSize = niiOriginal.shape
        sizeProcessing = (256,256,64)
        niiResized = resizeNii(pathNii=niiOriginal, newSize=sizeProcessing)
        niiMask = segmentLungs25D(niiResized, dirWithModel=self.pathModelLung,
                                  pathOutNii=None,
                                  outSize=inpSize,
                                  threshold=0.5)
        foutMskLung = '%s-msk-lung-test.nii.gz' % self.pathInpNii
        nib.save(niiMask, foutMskLung)
        self.assertTrue(inpSize==niiMask.shape)

    def test_lesion_segmentation(self):
        niiOriginal = nib.load(self.pathInpNii)
        inpSize = niiOriginal.shape
        sizeProcessing = (128, 128, 64)
        niiResized = resizeNii(pathNii=niiOriginal, newSize=sizeProcessing)
        niiMask = segmentLesions3D(niiResized, dirWithModel=self.pathModelLesion,
                                 pathOutNii=None,
                                 outSize=inpSize,
                                 threshold=None)
        foutMskLesion = '%s-msk-lesion-test.nii.gz' % self.pathInpNii
        nib.save(niiMask, foutMskLesion)
        self.assertTrue(inpSize == niiMask.shape)


if __name__ == '__main__':
    unittest.main()