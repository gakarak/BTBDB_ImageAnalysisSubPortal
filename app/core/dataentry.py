#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import json
import numpy as np
import nibabel as nib

class SeriesInfo:
    # (1) Study:
    studyID = None
    studyUID = None
    studyDate = None
    # (2) Series:
    pathNii  = None
    seriesUID = None
    jsonInfo = None
    modalityCode = None
    isGood = False
    shape = None
    def __init__(self, pathNii=None):
        if pathNii is not None:
            self.pathNii = pathNii
    def getKey(self):
        return '%s:%s' % (self.studyID, self.seriesUID)
    @staticmethod
    def getAllSeriesForStudy(caseDir, studyJson, isDropBad = False):
        studyID = studyJson['id']
        studyUID = studyJson['studyUid']
        studyDate = studyJson['studyDate']
        ret = dict()
        if 'series' in studyJson.keys():
            for ssi, seriesJson in enumerate(studyJson['series']):
                modalityCode = seriesJson['modality']['code']
                seriesUID = seriesJson['uid']
                pathNii = '%s/study-%s/series-%s-%s.nii.gz' % (caseDir, studyID, seriesUID, modalityCode)
                if os.path.isfile(pathNii):
                    # (1) Study:
                    pser = SeriesInfo(pathNii)
                    pser.studyID = studyID
                    pser.studyUID = studyUID
                    pser.studyDate = studyDate
                    # (2) Series:
                    pser.seriesUID = seriesUID
                    pser.modalityCode = modalityCode
                    pser.jsonInfo = seriesJson
                    skey = pser.getKey()
                    pser.isGood = SeriesInfo.isGoodSeries(pser)
                    if isDropBad and pser.isGood:
                        ret[skey] = pser
                else:
                    print ('*** WARNING *** cant find Nii-file [%s]' % pathNii)
        return ret
    @staticmethod
    def isGoodSeries(pseries):
        if pseries.modalityCode == 'CT' and os.path.isfile(pseries.pathNii):
            try:
                tmp = nib.load(pseries.pathNii)
                pseries.shape = np.array(tmp.shape)
                if len(tmp.shape)<3:
                    pseries.shape = np.array(list(tmp.shape) + [1])
                else:
                    pseries.shape = np.array(tmp.shape)
                if pseries.shape[2]>20:
                    return True
            except Exception as err:
                print ('**WARNING** cant read nii file [%s]' % err)
        return False

class CaseInfo:
    jsonShort = 'info-short.json'
    jsonAll = 'info-all.json'
    dictShort = None
    dictAll = None
    path = None
    #
    patientId = None
    caseId = None
    series = None
    isEmpty = True
    def __init__(self, path=None):
        if path is not None:
            self.loadInfo(path)
    def getKey(self):
        return '%s:%s' % (self.caseId, self.patientId)
    def loadInfo(self, pathCase, isDropBad = False):
        tpathJsonShort = os.path.join(pathCase, self.jsonShort)
        tpathJsonAll = os.path.join(pathCase, self.jsonAll)
        if os.path.isfile(tpathJsonShort):
            with open(tpathJsonShort, 'r') as f:
                self.jsonShort = json.load(f)
                self.patientId = self.jsonShort['patientId']
                self.caseId = self.jsonShort['conditionId']
        if os.path.isfile(tpathJsonAll):
            with open(tpathJsonAll, 'r') as f:
                self.jsonAll = json.load(f)
                if 'imagingStudies' in self.jsonAll.keys():
                    if self.jsonAll['imagingStudies'] is not None:
                        tdict = dict()
                        for idx, studyJson in enumerate(self.jsonAll['imagingStudies']):
                            dictSeries = SeriesInfo.getAllSeriesForStudy(pathCase, studyJson, isDropBad=isDropBad)
                            tdict = dict(tdict.items() + dictSeries.items())
                            # print ('Case [%s] : #Series = %d' % (self.caseId, len(self.series)))
                        if len(tdict)>0:
                            self.isEmpty = False
                        self.series = tdict

class DBWatcher:
    wdir = None
    cases = None
    def __init__(self, pdir = None):
        if pdir is not None:
            self.load(pdir)
    def load(self, pdir, isDropEmpty = False, isDropBadSeries=False):
        if not os.path.isdir(pdir):
            return
        self.wdir = pdir
        lstCases = glob.glob('%s/case-*' % self.wdir)
        numCases = len(lstCases)
        dictCases = dict()
        for ii, pathCase in enumerate(lstCases):
            caseInfo = CaseInfo()
            caseInfo.loadInfo(pathCase=pathCase, isDropBad=isDropBadSeries)
            if (ii%20 == 0):
                print ('[%d/%d] --> case #%s' % (ii, numCases, caseInfo.caseId))
            if isDropEmpty and caseInfo.isEmpty:
                continue
            else:
                tkey = caseInfo.getKey()
                dictCases[tkey] = caseInfo
        self.cases = dictCases
    def reload(self):
        if self.wdir is not None:
            self.load(self.wdir)
    def allSeries(self):
        for kcase, case in self.cases.items():
            if not case.isEmpty:
                for _, ser in case.series.items():
                    yield ser
    def printStat(self):
        if self.cases is not None:
            numCases = len(self.cases)
            numCasesEmpty = 0
            numSeries = 0
            numSeriesGood = 0
            for ic, (kcase, case) in enumerate(self.cases.items()):
                if case.isEmpty:
                    numCasesEmpty += 1
                else:
                    for si, (kstudy, ser) in enumerate(case.series.items()):
                        numSeries += 1
                        if ser.isGood:
                            numSeriesGood += 1
            print ('Cases: #All/#Empty = %d/%d, Series: All/Good = %d/%d'
                   % (numCases, numCasesEmpty, numSeries, numSeriesGood))
        else:
            print ('DBWatcher is not initialized')

if __name__ == '__main__':
    pass