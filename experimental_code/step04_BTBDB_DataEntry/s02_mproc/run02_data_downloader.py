#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import app
import app.backend
from app.core.utils.mproc import SimpleTaskManager
from app.core.utils.download import RunnerDBDownload

if __name__ == '__main__':
    # print(app.backend.config)
    # data_dir = '/home/ar/data/crdf/test_dataentry0'
    data_dir = app.backend.config.DIR_DATA
    runnerDataEntryDonwloader = RunnerDBDownload(data_dir=data_dir, limit=-1)
    runnerDataEntryDonwloader.refreshCases()
    print (runnerDataEntryDonwloader)

    runnerDataEntryDonwloader.run()

    # tmDataEntry = SimpleTaskManager(nproc=1, isThreadManager=True)
    # tmDataEntry.appendTaskRunner(runnerDataEntryDonwloader)
    # tmDataEntry.waitAll(dt=-1)
