#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'ar'

import os
from app.core.utils.cmd import RunnerDBConvert

#######################################
if __name__ == '__main__':
    dataDir = 'data-cases'
    runnerDBConvert = RunnerDBConvert(data_dir=dataDir)
    runnerDBConvert.run()
