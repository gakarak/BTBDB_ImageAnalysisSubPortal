#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar, snezhko'

import os
import time
import shutil
import numpy as np
import pandas as pd

from app.core.dataentry_v1 import DBWatcher
from app.core.segmct import api_generateColoredDICOM

if __name__ == '__main__':
    dataDir = '/media/data/datasets/@Data_BTBDB_ImageAnalysisSubPortal_debug_es'
    viewer_dir_root = dataDir + '_viewer'
    dbWatcher = DBWatcher()
    dbWatcher.load(dataDir, isDropEmpty=True, isDropBadSeries=True)
    dbWatcher.printStat()
    print(dbWatcher.toString())
    for ii, ser in enumerate(dbWatcher.allSeries()):
        if ser.isConverted():
            api_generateColoredDICOM(series=ser, viewer_dir_root=viewer_dir_root)

