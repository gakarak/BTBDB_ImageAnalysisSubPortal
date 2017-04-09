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

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

#######################################
dirData = 'data-cases'
# urlTakeList="http://tbportal-dataentry-dev.ibrsp.org/api/cases?since=2000-01-01&take=%d&skip=%d"
urlTakeList="http://tbportal-dataentry-dev.ibrsp.org/api/cases?since=2017-01-01&take=%d&skip=%d"
urlCaseInfo="http://tbportal-dataentry-dev.ibrsp.org/api/cases/%s"

# PATIENT_ID - CASE_ID - STUDY_ID - STUDY_UID - SERIES_UID - INSTANCE_UID
# urlDicomFile="http://data.tuberculosis.by/patient/%s/case/%s/imaging/study/%s/%s/series/%s/%s.dcm"
# urlDicomFile="http://tbportal-dataentry-dev.ibrsp.org/patient/%s/case/%s/imaging/study/%s/%s/series/%s/%s.dcm"
# urlDicomFile="http://tbportal-dataentry-prod.ibrsp.org/patient/%s/case/%s/imaging/study/%s/%s/series/%s/%s.dcm"
urlDicomFile="http://data.tbportals.niaid.nih.gov/patient/%s/case/%s/imaging/study/%s/%s/series/%s/%s.dcm"

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
    shutil.rmtree(dirData)
    mkdir_p(dirData)
    reqInfo = getListOfCases()
    numTotal = int(reqInfo['total'])
    for ii in range(numTotal):
        t0 = time.time()
        tretShort = getListOfCases(ptake=(ii + 1), pskip=ii)
        jsonInfoShort = tretShort['results'][0]
        # conditionId = jsonInfoShort['conditionId']
        conditionId = '2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0'
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
            for imageStudy in imageStudyInfo:
                tpatientId = jsonInfoCaseAll['patient']['id']
                tcaseId = jsonInfoCaseAll['id']
                timageStudyId = imageStudy['id']
                timageStudyUID = imageStudy['studyUid']
                for imageSeries in imageStudy['series']:
                    timageSeriesUID = imageSeries['uid']
                    tmodality = imageSeries['modality']
                    tnumInstances = imageSeries['numberOfInstances']
                    dirOutImageSeriesRaw = '%s/study-%s/series-%s/raw' % (dirOutCase, timageStudyId, timageSeriesUID)
                    mkdir_p(dirOutImageSeriesRaw)
                    for imageInstance in imageSeries['instance']:
                        tinstanceUID = imageInstance['uid']
                        instanceNumber = imageInstance['number']
                        currentDicomUrl = getDicomFileUrl(tpatientId,
                                                          tcaseId,
                                                          timageStudyId,
                                                          timageStudyUID,
                                                          timageSeriesUID, tinstanceUID)
                        foutDicom = '%s/instance-%04d.dcm' % (dirOutImageSeriesRaw, instanceNumber)
                        try:
                            data = downloadDicom(currentDicomUrl)
                        except Exception as err:
                            pass
                        print ('---')