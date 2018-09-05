#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import json
import argparse
import logging

from app.core.utils.mproc import SimpleTaskManager
from app.core.utils.download import RunnerDBDownload
from app.core.utils.cmd import RunnerDBConvert
from app.core.utils.report import RunnerMakeReport
import app.core.lesion_cbir as cbir

this_dir = os.path.dirname(os.path.realpath(__file__))
dir_models = os.path.join(this_dir, 'experimental_data', 'models')


################################################
class Config:
    def __init__(self, args):
        self.path_db = args.path_db
        self.threads = args.threads
        self.threads_download = args.threads_download
        if self.threads_download is None:
            self.threads_download = self.threads
        self.do_download = args.do_download
        self.do_convert  = args.do_convert
        self.do_process  = args.do_process
        self.do_cbir     = args.do_cbir
        #
        self.limit       = args.limit
    def to_json(self):
        return json.dumps(self.__dict__, indent=4)


################################################
def do_download(cfg:Config) -> bool:
    num_threads = cfg.threads_download
    runnerDataEntryDonwloader = RunnerDBDownload(data_dir=cfg.path_db, limit=cfg.limit)
    runnerDataEntryDonwloader.refreshCases()
    print(runnerDataEntryDonwloader)
    if (num_threads < 2):
        runnerDataEntryDonwloader.run()
    else:
        processor = SimpleTaskManager(nproc=num_threads, isThreadManager=True)
        processor.appendTaskRunner(runnerDataEntryDonwloader)
        processor.waitAll(dt=-1)
    return True


def do_convert(cfg:Config) -> bool:
    num_threads = cfg.threads
    runnerDBConvert = RunnerDBConvert(data_dir=cfg.path_db)
    if num_threads < 2:
        runnerDBConvert.run()
    else:
        processor = SimpleTaskManager(nproc=num_threads, isThreadManager=True)
        processor.appendTaskRunner(runnerDBConvert)
        processor.waitAll(dt=-1)
    return True


def do_process(cfg:Config) -> bool:
    path_model_lungs  = os.path.join(dir_models, 'fcnn_ct_lung_segm_2.5d_tf')
    path_model_lesion = os.path.join(dir_models, 'fcn3d_ct_lesion_segm_v3_tf')
    runnerMakeReport = RunnerMakeReport(data_dir=cfg.path_db,
                                        dirModelLung=path_model_lungs,
                                        dirModelLesion=path_model_lesion,
                                        listGpuId=[0])
    # FIXME: this runner can be runned acynchronously over mproc.SimpleTaskManager
    runnerMakeReport.run()
    return True


def do_cbir(cfg:Config) -> bool:
    num_threads = cfg.threads
    cbir.run_precalculate_lesion_dsc_for_db(cfg.path_db, num_threads)
    return True


################################################
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_db', type=str, required=True,  default=None, help='path to dataset (CRDF format)')
    parser.add_argument('--threads', type=int, required=False, default=1, help='#threads for processing')
    parser.add_argument('--threads_download', type=int, required=False, default=None, help='#threads for downloading')
    parser.add_argument('--do_download', action="store_true", help='run data downloading')
    parser.add_argument('--do_convert',  action="store_true", help='run data conversion')
    parser.add_argument('--do_process',  action="store_true", help='run data processing')
    parser.add_argument('--do_cbir',     action="store_true", help='run CBIR')
    #
    parser.add_argument('--limit', type=int, required=False, default=-1, help='limit for data downloading')
    args = parser.parse_args()
    logging.info('args = {}'.format(args))
    cfg = Config(args)
    logging.info('cfg = {}'.format(cfg.to_json()))
    #
    if cfg.do_download:
        do_download(cfg)
    if cfg.do_convert:
        do_convert(cfg)
    if cfg.do_process:
        do_process(cfg)
    if cfg.do_cbir:
        do_cbir(cfg)
