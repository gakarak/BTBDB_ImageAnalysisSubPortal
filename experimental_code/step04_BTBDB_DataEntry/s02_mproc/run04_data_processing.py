#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import app
import app.backend
from app.core.dataentry_v1 import DBWatcher
from app.core.utils.report import RunnerMakeReport

if __name__ == '__main__':
    # data_dir = 'data-real'
    data_dir = app.backend.config.DIR_DATA
    pathModelLung = '../../../experimental_data/models/fcnn_ct_lung_segm_2.5d_tf/'
    pathModelLesion = '../../../experimental_data/models/fcnn_ct_lesion_segm_3d_tf/'
    runnerMakeReport = RunnerMakeReport(data_dir=data_dir,
                                        dirModelLung=pathModelLung,
                                        dirModelLesion=pathModelLesion,
                                        listGpuId=[1])
    #FIXME: this runner can be runned acynchronously over mproc.SimpleTaskManager
    runnerMakeReport.run()
    print ('---')