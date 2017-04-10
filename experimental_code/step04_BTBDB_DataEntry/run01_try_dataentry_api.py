#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'ar'

import os
import time
import shutil
import requests
import json
from pprint import pprint
import errno

import logging
import logging.handlers

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

#######################################
dirData = 'data-cases'
# urlTakeList="http://tbportal-dataentry-dev.ibrsp.org/api/cases?since=2000-01-01&take=%d&skip=%d"

# urlTakeList="http://tbportal-dataentry-dev.ibrsp.org/api/cases?since=2000-02-01&take=%d&skip=%d"
# urlCaseInfo="http://tbportal-dataentry-dev.ibrsp.org/api/cases/%s"

urlTakeList="http://data.tbportals.niaid.nih.gov/api/cases?since=2000-02-01&take=%d&skip=%d"
urlCaseInfo="http://data.tbportals.niaid.nih.gov/api/cases/%s"

# PATIENT_ID - CASE_ID - STUDY_ID - STUDY_UID - SERIES_UID - INSTANCE_UID
# urlDicomFile="http://data.tuberculosis.by/patient/%s/case/%s/imaging/study/%s/%s/series/%s/%s.dcm"
# urlDicomFile="http://tbportal-dataentry-dev.ibrsp.org/patient/%s/case/%s/imaging/study/%s/%s/series/%s/%s.dcm"
# urlDicomFile="http://tbportal-dataentry-prod.ibrsp.org/patient/%s/case/%s/imaging/study/%s/%s/series/%s/%s.dcm"
urlDicomFile="http://data.tbportals.niaid.nih.gov/patient/%s/case/%s/imaging/study/%s/%s/series/%s/%s.dcm"
# urlDicomFile="http://data.tbportals.niaid.nih.gov/patient/%s/case/%s/imaging/study/%s/%s/series/%s/%s"

#######################################
def get_logger(wdir, logName=None):
    if logName is None:
        logName = "dicom-log-%s" % (time.strftime('%Y.%m.%d-%H.%M.%S'))
    else:
        logName = "%s-%s" % (logName, time.strftime('%Y.%m.%d-%H.%M.%S'))
    outLog = os.path.join(wdir, logName)
    logger = logging.getLogger(logName)
    logger.setLevel(logging.DEBUG)
    #
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    fh = logging.FileHandler("%s.log" % outLog)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

#######################################
def mkdir_p(path):
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

#######################################
def getDicomFileUrl(patientId, caseId, studyId, studyUID, seriesUID, instanceUID):
    return urlDicomFile % (patientId, caseId, studyId, studyUID, seriesUID, instanceUID)

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
if __name__ == '__main__':
    # if os.path.isdir(dirData):
    #     shutil.rmtree(dirData)
    mkdir_p(dirData)
    reqInfo = getListOfCases()
    numTotal = int(reqInfo['total'])
    ptrLogger = get_logger(wdir=dirData)
    for ii in range(numTotal):
        t0 = time.time()
        tretShort = getListOfCases(ptake=(ii + 1), pskip=ii)
        jsonInfoShort = tretShort['results'][0]
        conditionId = jsonInfoShort['conditionId']
        # conditionId = '2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0'
        # conditionId = '3602e1da-03b0-416c-9aed-57f16d5cd5fc'
        jsonInfoCaseAll = getCaseInfo(condId=conditionId)
        imageStudyInfo = jsonInfoCaseAll['imagingStudies']
        dt = time.time() - t0
        print ('[%d/%d] --> (%s), dt = %0.3f(s)' % (ii, numTotal, conditionId, dt))
        dirOutCase = os.path.join(dirData, 'case-%s' % conditionId)
        mkdir_p(dirOutCase)
        #
        foutJsonInfoShort = '%s/info-short.json' % dirOutCase
        foutJsonInfoLong  = '%s/info-all.json' % dirOutCase
        with open(foutJsonInfoShort, 'w') as f:
            f.write(json.dumps(jsonInfoShort, indent=4))
        with open(foutJsonInfoLong, 'w') as f:
            f.write(json.dumps(jsonInfoCaseAll, indent=4))
        if imageStudyInfo is None:
            print ('\t*** no image studies')
        else:
            print ('\t#ImageStudies = %d' % len(imageStudyInfo))
            numStudy = len(imageStudyInfo)
            for iiStudy, imageStudy in enumerate(imageStudyInfo):
                tpatientId = jsonInfoCaseAll['patient']['id']
                tcaseId = jsonInfoCaseAll['id']
                timageStudyId = imageStudy['id']
                timageStudyUID = imageStudy['studyUid']
                numSeries = len(imageStudy['series'])
                for iiSeries, imageSeries in enumerate(imageStudy['series']):
                    T0 = time.time()
                    timageSeriesUID = imageSeries['uid']
                    tmodality = imageSeries['modality']['code']
                    numInstances = imageSeries['numberOfInstances']
                    dirOutImageSeriesRaw = '%s/study-%s/series-%s/raw' % (dirOutCase, timageStudyId, timageSeriesUID)
                    mkdir_p(dirOutImageSeriesRaw)
                    numI = len(imageSeries['instance'])
                    # isError = False
                    print('\t\t[%d/%d * %d/%d] #Series = %d ...' % (iiStudy, numStudy, iiSeries, numSeries, numInstances), end='')
                    for imageInstance in imageSeries['instance']:
                        tinstanceUID = imageInstance['uid']
                        try:
                            instanceNumber = imageInstance['number']
                        except Exception as err:
                            # ptrLogger.error("**ERROR** invalid imageInstance structure for [%s] : %s" % (dirOutImageSeriesRaw, err))
                            # break
                            instanceNumber = 0
                        currentDicomUrl = getDicomFileUrl(tpatientId,
                                                          tcaseId,
                                                          timageStudyId,
                                                          timageStudyUID,
                                                          timageSeriesUID, tinstanceUID)
                        try:
                            foutDicom = '%s/instance-%s-%04d.dcm' % (dirOutImageSeriesRaw, tmodality, instanceNumber)
                            if os.path.isfile(foutDicom):
                                ptrLogger.info("***WARNING*** file exist [%s], skip..." % foutDicom)
                            data = downloadDicom(currentDicomUrl)
                        except Exception as err:
                            ptrLogger.error("**ERROR** cant read url [%s] : %s" % (currentDicomUrl, err))
                            break
                        with open(foutDicom, 'wb') as f:
                            f.write(data.getvalue())
                        # print ('---')
                    dT = time.time() - T0
                    print (' ... dT ~ %0.3f (s)' % dT)
                    # print ('\t\t[%d/%d * %d/%d] : dT ~ %0.3f (s)' % (iiStudy, numStudy, iiSeries, iiSeries, dT))
