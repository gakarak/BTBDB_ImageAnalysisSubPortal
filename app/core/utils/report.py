#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
from datetime import datetime
from app.core.dataentry_v1 import DBWatcher
import app.core.utils.log as log
import app.core.segmct as segm

import app.core.utils.mproc as mproc

class TaskRunnerMakeReport(mproc.AbstractRunner):
    def __init__(self, series, dirModelLung, dirModelLesion, gpuId=None):
        self.series = series
        self.dir_model_lung = dirModelLung
        self.dir_model_lesion = dirModelLesion
        self.gpuId = gpuId
    def getUniqueKey(self):
        return self.series.getKey()
    def run(self):
        if self.gpuId is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = '{0}'.format(self.gpuId)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '' # on CPU
        wdir = self.series.getDir(isRelative=False)
        if os.path.isdir(wdir):
            ptrLogger = log.get_logger(wdir=wdir, logName='s03-report')
            tret = segm.api_generateAllReports(series=self.series,
                                        dirModelLung=self.dir_model_lung,
                                        dirModelLesion=self.dir_model_lesion,
                                        ptrLogger=ptrLogger)
            ptrLogger.info('MakeReport: isOk={0}'.format(tret))


class RunnerMakeReport(mproc.AbstractRunner):
    def __init__(self, dirModelLung, dirModelLesion, data_dir=None, listGpuId = []):
        if data_dir is None:
            #FIXME: remove in future
            self.data_dir = 'data-cases'
        else:
            self.data_dir = data_dir
        self.dir_model_lung     = dirModelLung
        self.dir_model_lesion   = dirModelLesion
        if (listGpuId is not None) and (len(listGpuId)<1):
            self.listGpuId = None
        else:
            self.listGpuId = listGpuId
    def getUniqueKey(self):
        return 'report-tkey-{0}'.format(datetime.now().strftime('%Y.%m.%d-%H.%M.%S:%f'))
    def run(self, tm=None):
        dirData = self.data_dir
        if os.path.isdir(dirData):
            ptrLogger = log.get_logger(wdir=dirData, logName='s02-report')
            dbWatcher = DBWatcher(pdir=dirData)
            ptrLogger.info (dbWatcher.toString())
            if self.listGpuId is not None:
                arrGpuId = np.array(self.listGpuId)
            else:
                arrGpuId = None
            for iser, ser in enumerate(dbWatcher.allSeries()):
                if ser.isConverted():
                    if ser.isPostprocessed():
                        ptrLogger.info('[{0}] *** Series processe, skip ...{0}'.format(iser, ser))
                        continue
                    if arrGpuId is not None:
                        pgpuId = int(arrGpuId[0])
                    else:
                        pgpuId = None
                    taskMakeReport = TaskRunnerMakeReport(series=ser,
                                                          dirModelLung=self.dir_model_lung,
                                                          dirModelLesion=self.dir_model_lesion,
                                                          gpuId=pgpuId)
                    arrGpuId = np.roll(arrGpuId, 1)
                    if tm is None:
                        tret = taskMakeReport.run()
                        ptrLogger.info('[{0}] makereport isOk={1}, {2}'.format(iser, tret, ser))
                    else:
                        tm.appendTaskRunner(taskMakeReport)
                        ptrLogger.info('{0} : add report generation in queue : {1}'.format(iser, ser))
            dbWatcher.reload()
            ptrLogger.info(dbWatcher.toString())

if __name__ == '__main__':
    pass