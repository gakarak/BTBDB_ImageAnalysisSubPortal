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

from app.core.dataentry_v1 import DBWatcher, CaseInfo, SeriesInfo
from . import mkdir_p
from . import log

from . import mproc
from datetime import datetime

urlTakeList="https://data.tbportals.niaid.nih.gov/api/cases?since=2010-04-01&take=%d&skip=%d"
urlCaseInfo="https://data.tbportals.niaid.nih.gov/api/cases/%s"

#######################################
try:
    from cStringIO import StringIO
except:
    import io as StringIO
    # from StringIO import StringIO

#######################################
def processRequest(urlRequest):
    tret = requests.get(urlRequest)
    if tret.status_code == 200:
        # return json.loads(tret._content)
        return tret.json()
    else:
        strErr = 'Error: %s' % tret._content
        print('*** ERROR: %s' % urlRequest)
        # pprint(json.loads(tret._content))
        pprint(tret.json())
        raise Exception(strErr)

def getListOfCases(ptake=1, pskip=0):
    if ptake > 1000:
        print('!!!Warning!!! #requested items is reduced by {}'.format(ptake))
        ptake = 1000
    urlRequest = urlTakeList % (ptake, pskip)
    return processRequest(urlRequest)

def getCaseInfo(condId):
    urlRequest = urlCaseInfo % condId
    return processRequest(urlRequest)

def downloadDicom(urlRequest, pauthTocken=None):
    tret = requests.get(urlRequest, auth=pauthTocken, stream=True)
    if tret.status_code == 200:
        buff = StringIO()
        for chunk in tret.iter_content(2048):
            buff.write(chunk)
        return buff
    else:
        strErr = 'Error: %s' % tret._content
        print('*** ERROR: %s' % urlRequest)
        pprint(json.loads(tret._content))
        raise Exception(strErr)

#######################################
class TaskRunnerDownloadSeries(mproc.AbstractRunner):
    def __init__(self, series, isCleanBeforeStart = False):
        self.series = series
        self.is_clean = isCleanBeforeStart
    def getUniqueKey(self):
        return self.series.getKey()
    def run(self):
        if self.series.isInitialized():
            wdir = self.series.getDir(isRelative=False)
            wdirRaw = self.series.getDirRaw(isRelative=False)
            if self.series.isHasData():
                if self.is_clean:
                    if os.path.isdir(wdir):
                        shutil.rmtree(wdir)
                mkdir_p(wdir)
                mkdir_p(wdirRaw)
                ptrLogger = log.get_logger(wdir=wdir, logName='stage1_donwload')
                ptrLogger.info("start downloading : {0}".format(self.getUniqueKey()))
                lstInstJs = self.series.getInstancesJs()
                for instJs in lstInstJs:
                    foutDicom = os.path.join(wdirRaw, self.series.getInstanceBaseNameJs(instJs))
                    # FIXME: in future you need check downloaded file size to more accurate validation of "existing" DICOM
                    if not os.path.isfile(foutDicom):
                        ptrLogger.info('start downloading [{0}]'.format(foutDicom))
                        try:
                            urlDicom = instJs['content']['url']
                            data = downloadDicom(urlDicom)
                            with open(foutDicom, 'wb') as f:
                                f.write(data.getvalue())
                        except Exception as err:
                            ptrLogger.error('Cant download DICOM URL for instance [{0}] : {1}'.format(instJs, err))
                    else:
                        ptrLogger.info('*** WARNING *** file [{0}] exist, skip'.format(foutDicom))

#######################################
class RunnerDBDownload(mproc.AbstractRunner):
    def __init__(self, data_dir=None):
        if data_dir is None:
            # FIXME: remove in future
            self.data_dir = 'data-cases'
        else:
            self.data_dir = data_dir
        self.clean()
    def clean(self):
        self.allCases = None
    def numCases(self):
        if self.allCases is not None:
            return len(self.allCases)
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
        tmp = getListOfCases(1, 0)
        numCases = tmp['total']
        tmp = getListOfCases(numCases, 0)
        self.allCases = tmp['results']
    # AbstractRunner interface:
    def getUniqueKey(self):
        return 'download-tkey-{0}'.format(datetime.now().strftime('%Y.%m.%d-%H.%M.%S:%f'))
    def run(self):
        dirData = self.data_dir
        mkdir_p(dirData)
        ptrLogger = log.get_logger(wdir=dirData, logName='s01-donwload')
        dbWatcher = DBWatcher(pdir=dirData)
        ptrLogger.info(dbWatcher.toString())
        #
        # dataEntry = DataEntryRunner()
        try:
            self.refreshCases()
        except Exception as err:
            ptrLogger.error('Cant refresh info about DataEntry cases: {0}'.format(err))
        ptrLogger.info ('DataEntry: %s' % self.toString())
        numCases = len(self.allCases)
        for icase, case in enumerate(self.allCases):
            caseId = case['conditionId']
            ptrLogger.info ('[%d/%d] ' % (icase, numCases))
            if dbWatcher.isHaveCase(caseId):
                new_case = dbWatcher.cases[caseId]
                ptrLogger.info('\t:: case-info [%s] exist in db-cache, skip get-case-info ...' % caseId)
            else:
                try:
                    caseInfo = getCaseInfo(caseId)
                    isSaveToDisk = True
                    new_case = CaseInfo.newCase(dataDir=dirData,
                                                dictShort=case,
                                                dictAll=caseInfo,
                                                isSaveToDisk=isSaveToDisk)
                except Exception as err:
                    ptrLogger.error('Error case download: [{0}]'.format(err))
                    new_case = None
            if new_case is not None:
                new_series = new_case.getGoodSeries()
                # try:
                if len(new_series) > 0:
                    for ser in new_series:
                        if ser.isInitialized() and ser.isHasData() and (not ser.isDownloaded()):
                            if ser.isConverted():
                                ptrLogger.warn('\t Series converted (exist nii) but is not downloaded: skip...')
                            else:
                                # FIXME: execute SeriesDownload Runner over Process-TaskManager in future
                                ptrLogger.info('\t append Series to download [{0}]'.format(ser))
                                newRunner = TaskRunnerDownloadSeries(series=ser)
                                newRunner.run()
                        else:
                            ptrLogger.info('\tskip downloaded series: {0}'.format(ser))

#######################################
if __name__ == '__main__':
    pass