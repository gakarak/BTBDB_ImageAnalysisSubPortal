#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import app.core.utils as apput
from app.core.utils import download as dwd
import app.core.utils.log as log
from app.core.dataentry_v1 import DBWatcher, CaseInfo, SeriesInfo

class DataEntry:
    def __init__(self):
        self.clean()
    def clean(self):
        self.allCases = None
    def toString(self):
        if self.allCases is None:
            return 'No cases'
        else:
            return '#Cases = %d' % len(self.allCases)
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def refreshCases(self):
        tmp = dwd.getListOfCases(1, 0)
        numCases = tmp['total']
        tmp = dwd.getListOfCases(numCases, 0)
        self.allCases = tmp['results']


if __name__ == '__main__':
    #
    dirData = 'data-cases'
    ptrLogger = log.get_logger(wdir=dirData)
    apput.mkdir_p(dirData)
    dbWatcher = DBWatcher(pdir=dirData)
    dbWatcher.printStat()
    #
    dataEntry = DataEntry()
    dataEntry.refreshCases()
    print ('DataEntry: %s' % dataEntry)
    numCases = len(dataEntry.allCases)
    for icase, case in enumerate(dataEntry.allCases):
        caseId = case['conditionId']
        print ('[%d/%d]' % (icase, numCases))
        if dbWatcher.isHaveCase(caseId):
            isSaveToDisk = False
            print ('\t:: case [%s] exist, skip...' % caseId)
        else:
            isSaveToDisk = True
        try:
            caseInfo = dwd.getCaseInfo(caseId)
            new_case = CaseInfo.newCase(dataDir=dirData,
                                        dictShort=case,
                                        dictAll=caseInfo,
                                        isSaveToDisk=isSaveToDisk)
            new_series = new_case.getGoodSeries()
            if len(new_series)>0:
                for ser in new_series:
                    if not dbWatcher.checkSeriesInDB(ser):
                        print ('\t append Series to download [{0}]'.format(ser))
                    else:
                        print ('\tskip existing series: {0}'.format(ser))
                print ('---')
        except Exception as err:
            ptrLogger.error('Error case download: [{0}]'.format(err))

