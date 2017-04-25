#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'ar'

import app.core.preprocessing as dproc
import app.core.utils.mproc as mproc
from app.core.dataentry_v1 import DBWatcher

if __name__ == '__main__':
    dataDir = 'data-cases'
    dbWatcher = DBWatcher(pdir=dataDir)
    dbWatcher.printStat()
