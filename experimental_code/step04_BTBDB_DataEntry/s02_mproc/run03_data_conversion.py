#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'ar'

import os
import app
import app.backend
from app.core.utils.cmd import RunnerDBConvert
from app.core.utils.mproc import SimpleTaskManager
import argparse
import logging

#######################################
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_db', type=str, required=False, default=None, help='path to dataset (CRDF format)')
    parser.add_argument('--threads', type=int, required=False, default=1, help='#threads for CBIR processing')
    args = parser.parse_args()
    logging.info('args = {}'.format(args))
    if args.path_db is None:
        db_dir = app.backend.config.DIR_DATA
    else:
        db_dir = args.path_db
    # data_dir = 'data-cases'
    data_dir = os.path.basename(app.backend.config.DIR_DATA)
    runnerDBConvert = RunnerDBConvert(data_dir=data_dir)
    if args.threads < 2:
        runnerDBConvert.run()
    else:
        processor = SimpleTaskManager(nproc=args.threads, isThreadManager=True)
        processor.appendTaskRunner(runnerDBConvert)
        processor.waitAll(dt=-1)
