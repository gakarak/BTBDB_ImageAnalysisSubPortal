#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import app.backend
import logging
import app.core.lesion_cbir as cbir
import argparse


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
    cbir.run_precalculate_lesion_dsc_for_db(db_dir, args.threads)
