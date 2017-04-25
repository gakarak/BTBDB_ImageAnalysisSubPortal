#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

from app.core.utils.mproc import SimpleTaskManager
from app.core.utils.download import RunnerDBDownload

if __name__ == '__main__':
    runnerDataEntry = RunnerDBDownload()
    runnerDataEntry.refreshCases()
    print (runnerDataEntry)

    # runnerDataEntry.run()

    tmDataEntry = SimpleTaskManager(nproc=1, isThreadManager=True)
    tmDataEntry.appendTaskRunner(runnerDataEntry)
    tmDataEntry.waitAll(dt=-1)
