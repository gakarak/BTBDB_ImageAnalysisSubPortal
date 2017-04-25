#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

from app.core.utils.download import RunnerDataEntry

if __name__ == '__main__':
    runnerDataEntry = RunnerDataEntry()
    runnerDataEntry.refreshCases()
    print (runnerDataEntry)
