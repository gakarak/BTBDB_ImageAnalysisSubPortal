#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'ar'

import os
import app.core.preprocessing as dproc
import app.core.utils.mproc as mproc
from app.core.utils.cmd import pydcm2nii
from app.core.dataentry_v1 import DBWatcher

class TaskRunnerConvertSeries(mproc.AbstractRunner):
    def __init__(self, series):
        self.ser = series
    def run(self):
        pass


if __name__ == '__main__':
    dataDir = 'data-cases'
    dbWatcher = DBWatcher(pdir=dataDir)
    print (dbWatcher.toString())
    for iser, ser in enumerate(dbWatcher.allSeries()):
        #FIXME: we think, tha if series is postprocessed, then series-conversion is not needed...
        if ser.isDownloaded() and (not ser.isConverted()) and (not ser.isPostprocessed()):
            inpDirWithDicom = ser.getDirRaw(False)
            outPathNifti = ser.pathConvertedNifti(False)
            tret = pydcm2nii(dirDicom=inpDirWithDicom, foutNii=outPathNifti)
            print ('[%d] conver is Ok = %s, %s' % (iser, tret, ser))
    # dbWatcher.printStat()
