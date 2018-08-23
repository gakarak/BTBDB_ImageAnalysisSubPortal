#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import app
import logging
import app.backend
import argparse
import app.core.utils as utils
from app.core.dataentry_v1 import DBWatcher
from app.core.utils.report import RunnerMakeReport

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_db', type=str, required=False, default=None, help='path to dataset (CRDF format)')
    args = parser.parse_args()
    logging.info('args = {}'.format(args))
    #
    # data_dir = 'data-real'
    if args.path_db is None:
        db_dir = app.backend.config.DIR_DATA
    else:
        db_dir = args.path_db
    if not os.path.isdir(db_dir):
        raise FileNotFoundError('Cant find db-directory: [{}]'.format(db_dir))
    data_dir = utils.get_data_dir()
    pathModelLung   = os.path.join(data_dir, 'models', 'fcnn_ct_lung_segm_2.5d_tf')
    pathModelLesion = os.path.join(data_dir, 'models', 'fcn3d_ct_lesion_segm_v3_tf')
    print('-')
    # pathModelLung = '../../../experimental_data/models/fcnn_ct_lung_segm_2.5d_tf/'
    # pathModelLesion = '../../../experimental_data/models/fcnn_ct_lesion_segm_3d_tf/'
    # pathModelLesion = '../../../experimental_data/models/fcn3d_ct_lesion_segm_v3_tf/'
    runnerMakeReport = RunnerMakeReport(data_dir=db_dir,
                                        dirModelLung=pathModelLung,
                                        dirModelLesion=pathModelLesion,
                                        listGpuId=[0])
    #FIXME: this runner can be runned acynchronously over mproc.SimpleTaskManager
    runnerMakeReport.run()
    print ('---')