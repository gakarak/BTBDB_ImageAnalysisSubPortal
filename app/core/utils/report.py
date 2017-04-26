#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
from datetime import datetime
from app.core.dataentry_v1 import DBWatcher
import app.core.utils.log as log
import app.core.segmct as segm

import app.core.utils.mproc as mproc

class TaskRunnerMakeReport(mproc.AbstractRunner):
    def __init__(self, series, dirModelLung, dirModelLesion):
        self.series = series
        self.dir_model_lung = dirModelLung
        self.dir_model_lesion = dirModelLesion
    def getUniqueKey(self):
        return self.series.getKey()
    def run(self):
        wdir = self.series.getDir(isRelative=False)
        if os.path.isdir(wdir):
            ptrLogger = log.get_logger(wdir=wdir, logName='s03-report')
            tret = segm.api_generateAllReports(series=self.series,
                                        dirModelLung=self.dir_model_lung,
                                        dirModelLesion=self.dir_model_lesion,
                                        ptrLogger=ptrLogger)
            ptrLogger.info('MakeReport: isOk={0}'.format(tret))


class RunnerMakeReport(mproc.AbstractRunner):
    def __init__(self, dirModelLung, dirModelLesion, data_dir=None):
        if data_dir is None:
            #FIXME: remove in future
            self.data_dir = 'data-cases'
        else:
            self.data_dir = data_dir
        self.dir_model_lung     = dirModelLung
        self.dir_model_lesion   = dirModelLesion
    def getUniqueKey(self):
        return 'report-tkey-{0}'.format(datetime.now().strftime('%Y.%m.%d-%H.%M.%S:%f'))
    def run(self):
        dirData = self.data_dir
        if os.path.isdir(dirData):
            ptrLogger = log.get_logger(wdir=dirData, logName='s02-report')
            dbWatcher = DBWatcher(pdir=dirData)
            ptrLogger.info (dbWatcher.toString())
            for iser, ser in enumerate(dbWatcher.allSeries()):
                if ser.isConverted():
                    taskMakeReport = TaskRunnerMakeReport(series=ser,
                                                          dirModelLung=self.dir_model_lung,
                                                          dirModelLesion=self.dir_model_lesion)
                    tret = taskMakeReport.run()
                    ptrLogger.info('[{0}] makereport isOk={1}, {2}'.format(iser, tret, ser))
            dbWatcher.reload()
            ptrLogger.info(dbWatcher.toString())

if __name__ == '__main__':
    pass