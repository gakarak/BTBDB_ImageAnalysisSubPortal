#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'ar'

import os
import app
import app.backend
from app.core.utils.cmd import RunnerDBConvert

#######################################
if __name__ == '__main__':
    # data_dir = 'data-cases'
    data_dir = os.path.basename(app.backend.config.DIR_DATA)
    runnerDBConvert = RunnerDBConvert(data_dir=data_dir)
    runnerDBConvert.run()
