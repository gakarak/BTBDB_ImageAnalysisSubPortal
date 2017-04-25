#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import shutil
import requests
import json
from pprint import pprint
import errno

import logging
import logging.handlers

#######################################
def get_logger(wdir, logName=None, isDefault=False):
    if isDefault:
        logName = 'default.log'
    else:
        if logName is None:
            logName = "dicom-log-%s" % (time.strftime('%Y.%m.%d-%H.%M.%S'))
        else:
            logName = "%s-%s" % (logName, time.strftime('%Y.%m.%d-%H.%M.%S'))
    outLog = os.path.join(wdir, logName)
    logger = logging.getLogger(logName)
    logger.setLevel(logging.DEBUG)
    #
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    fh = logging.FileHandler("%s.log" % outLog, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

#######################################
if __name__ == '__main__':
    pass