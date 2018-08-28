#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import app
import app.backend
from app.core.utils.mproc import SimpleTaskManager
from app.core.utils.download import RunnerDBDownload
import argparse
import logging


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_db', type=str, required=False, default=None, help='path to dataset (CRDF format)')
    parser.add_argument('--threads', type=int, required=False, default=1, help='#threads for downloading')
    args = parser.parse_args()
    logging.info('args = {}'.format(args))
    if args.path_db is None:
        db_dir = app.backend.config.DIR_DATA
    else:
        db_dir = args.path_db
    #
    runnerDataEntryDonwloader = RunnerDBDownload(data_dir=db_dir, limit=-1)
    runnerDataEntryDonwloader.refreshCases()
    print (runnerDataEntryDonwloader)
    if (args.threads < 2):
        runnerDataEntryDonwloader.run()
    else:
        processor = SimpleTaskManager(nproc=args.threads, isThreadManager=True)
        processor.appendTaskRunner(runnerDataEntryDonwloader)
        processor.waitAll(dt=-1)
