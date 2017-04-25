#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import nibabel as nib
import matplotlib.pyplot as plt
from app.core.preprocessing import resizeNii
from app.core.segmct import segmentLesions3D, segmentLungs25D
from app.core.dataentry_v1 import DBWatcher
from app.core.segmct import api_segmentLungAndLesion
# import common as comm

if __name__ == '__main__':
    pathModelLung = '../../experimental_data/models/fcnn_ct_lung_segm_2.5d_tf/'
    pathModelLesion ='../../experimental_data/models/fcnn_ct_lesion_segm_3d_tf/'
    #
    # shape4Lung = (256, 256, 64)
    # shape4Lesi = (128, 128, 64)
    dataDir = '../../experimental_data/dataentry_test0'
    dbWatcher = DBWatcher(pdir=dataDir)
    dbWatcher.load(dataDir, isDropEmpty=True, isDropBadSeries=True)
    # dbWatcher.printStat()
    print (dbWatcher.toString())
    for ii, ser in enumerate(dbWatcher.allSeries()):
        if ser.isConverted():
            api_segmentLungAndLesion(dirModelLung=pathModelLung,
                                     dirModelLesion=pathModelLesion,
                                     series=ser,
                                     ptrLogger=None)
            pathSegmLesion = ser.pathPostprocLesions(isRelative=False)
            print ('[{0}] : isOk={2}, {1}'.format(ii, ser, os.path.isfile(pathSegmLesion)))
        # dataNii = nib.load(ser.pathNii)
        # shapeOrig = dataNii.shape
        # niiResiz4Lung = resizeNii(dataNii, shape4Lung)
        # niiResiz4Lesi = resizeNii(dataNii, shape4Lesi)
        # #
        # lungMask = segmentLungs25D(niiResiz4Lung,
        #                            dirWithModel=pathModelLung,
        #                            pathOutNii=None,
        #                            outSize=shapeOrig,
        #                            # outSize=shape4Lung,
        #                            threshold=0.5)
        # lesionMask = segmentLesions3D(niiResiz4Lesi,
        #                               dirWithModel=pathModelLesion,
        #                               pathOutNii=None,
        #                               outSize=shapeOrig,
        #                               # outSize=shape4Lung,
        #                               threshold=None)
        # foutLungMsk = '%s-lungs.nii.gz' % ser.pathNii
        # foutLesionMsk = '%s-lesion.nii.gz' % ser.pathNii
        # nib.save(lungMask, foutLungMsk)
        # nib.save(lesionMask, foutLesionMsk)
        # print ('[%d] --> [%s]' % (ii, foutLesionMsk))