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

from . import mkdir_p
import log

import mproc

urlTakeList="https://data.tbportals.niaid.nih.gov/api/cases?since=2017-02-01&take=%d&skip=%d"
urlCaseInfo="https://data.tbportals.niaid.nih.gov/api/cases/%s"

#######################################
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

#######################################
def processRequest(urlRequest):
    tret = requests.get(urlRequest)
    if tret.status_code == 200:
        return json.loads(tret._content)
    else:
        strErr = 'Error: %s' % tret._content
        print('*** ERROR: %s' % urlRequest)
        pprint(json.loads(tret._content))
        raise Exception(strErr)

def getListOfCases(ptake=1, pskip=0):
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
class RunnerDownloadSeries(mproc.AbstractRunner):
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
if __name__ == '__main__':
    pass