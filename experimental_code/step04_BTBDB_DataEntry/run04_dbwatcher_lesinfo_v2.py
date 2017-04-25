#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

from app.core.dataentry_v1 import DBWatcher
import app.core.segmct as segm

###############################
if __name__ == '__main__':
    pathModelLung = '../../experimental_data/models/fcnn_ct_lung_segm_2.5d_tf/'
    pathModelLesion = '../../experimental_data/models/fcnn_ct_lesion_segm_3d_tf/'
    dataDir = '../../experimental_data/dataentry_test0'
    dbWatcher = DBWatcher(pdir=dataDir)
    dbWatcher.load(dataDir, isDropEmpty=True, isDropBadSeries=True)
    print (dbWatcher.toString())
    for ii, ser in enumerate(dbWatcher.allSeries()):
        if ser.isConverted():
            # (0) prepare path-variables
            tret = segm.api_generateAllReports(series=ser,
                                        dirModelLung=pathModelLung,
                                        dirModelLesion=pathModelLesion,
                                        ptrLogger=None)
            print ('[{0}] report isOk={1} for {2}'.format(ii, tret, ser))
