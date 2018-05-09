#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

from app.core.utils.mproc import SimpleTaskManager
from app.core.utils.download import RunnerDBDownload

if __name__ == '__main__':
    data_dir = '/home/ar/data/crdf/test_dataentry0'
    runnerDataEntryDonwloader = RunnerDBDownload(data_dir=data_dir, limit=2)
    runnerDataEntryDonwloader.refreshCases()
    print (runnerDataEntryDonwloader)

    runnerDataEntryDonwloader.run()

    # tmDataEntry = SimpleTaskManager(nproc=1, isThreadManager=True)
    # tmDataEntry.appendTaskRunner(runnerDataEntryDonwloader)
    # tmDataEntry.waitAll(dt=-1)
