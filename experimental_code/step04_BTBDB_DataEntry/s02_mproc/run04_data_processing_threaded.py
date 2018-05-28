#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import app.backend
from app.core.dataentry_v1 import DBWatcher
from app.core.utils.mproc import SimpleTaskManager
from app.core.utils.report import RunnerMakeReport

import os
from app.core.dataentry_v1 import DBWatcher
from app.core.segmct import api_segmentLungAndLesion


if __name__ == '__main__':
    # dataDir = 'data-cases'
    data_dir = app.backend.config.DIR_DATA
    pathModelLung = '../../../experimental_data/models/fcnn_ct_lung_segm_2.5d_tf/'
    pathModelLesion = '../../../experimental_data/models/fcnn_ct_lesion_segm_3d_tf/'

    # dbWatcher = DBWatcher(pdir=dataDir)
    # for iser, ser in enumerate(dbWatcher.allSeries()):
    #     if ser.isConverted() and (not ser.isPostprocessed()):
    #         pathNii = ser.pathConvertedNifti(isRelative=False)
    #         print (os.path.dirname(pathNii))
    #         os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #         api_segmentLungAndLesion(dirModelLung=pathModelLung,
    #                                  dirModelLesion=pathModelLesion,
    #                                  series=ser,
    #                                  gpuMemUsage=0.2)
    #         print ('---')

    runnerMakeReport = RunnerMakeReport(data_dir=data_dir,
                                        dirModelLung=pathModelLung,
                                        dirModelLesion=pathModelLesion,
                                        listGpuId=[0, 1])
    #FIXME: this runner can be runned acynchronously over mproc.SimpleTaskManager
    tm = SimpleTaskManager(nproc=4, isThreadManager=False)
    runnerMakeReport.run(tm=tm)
    tm.waitAll()
    print ('---')