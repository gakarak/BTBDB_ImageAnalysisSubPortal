#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures as mp
import logging
from logging import getLogger, basicConfig

from app.core.dataentry_v1 import DBWatcher
from app.core.segmct import api_generateColoredDICOM


logger = getLogger('dicom-coloring')
basicConfig(level=logging.INFO)
logger.setLevel(logging.DEBUG)


def task_dicom_coloring(pdata):
    idx, idx_num, [series_copy, dir_color_dicom] = pdata
    t1 = time.time()
    logger.info('\t[{}/{}] start DICOM series coloring [{}]'.format(idx, idx_num, series_copy))
    api_generateColoredDICOM(series_copy, viewer_dir_root=dir_color_dicom)
    dt = time.time() - t1
    logger.info('\t\t[{}/{}] ... dt ~ {:0.2f} (s), [{}]'.format(idx, idx_num, dt, series_copy))


def process_color_dicom_for_db(dir_db, dir_color_dicom, max_threads=1):
    if not os.path.isdir(dir_db):
        raise FileNotFoundError('Cant find DB-directory [{}]'.format(dir_db))
    os.makedirs(dir_color_dicom, exist_ok=True)
    list_data_tasks = []
    dbWatcher = DBWatcher(pdir=dir_db)
    # logger.info(dbWatcher)
    for iser, series in enumerate(dbWatcher.allSeries()):
        if series.isPostprocessed():
            list_data_tasks.append([copy.copy(series), dir_color_dicom])
        else:
            logger.warning('\t!!! series [{}] is not preprocessed!'.format(series))
    num_tasks = len(list_data_tasks)
    list_data_tasks = [[ii, num_tasks, xx] for ii, xx in enumerate(list_data_tasks)]
    if (max_threads < 2) or (num_tasks < 2):
        ret = [task_dicom_coloring(xx) for xx in list_data_tasks]
    else:
        pool = mp.ProcessPoolExecutor(max_workers=max_threads)
        ret = list(pool.map(task_dicom_coloring, list_data_tasks))
        pool.shutdown(wait=True)
    return True


if __name__ == '__main__':
    pass