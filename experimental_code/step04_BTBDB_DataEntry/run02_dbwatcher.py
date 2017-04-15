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
    def __init__(self, pathNii=None):
        if pathNii is not None:
            self.pathNii = pathNii
    def getKey(self):
        return '%s:%s' % (self.studyID, self.seriesUID)
    @staticmethod
    def getAllSeriesForStudy(caseDir, studyJson):
        studyID = studyJson['id']
        studyUID = studyJson['studyUid']
        studyDate = studyJson['studyDate']
        ret = {}
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
                    ret[skey] = pser
                else:
                    print ('*** WARNING *** cant find Nii-file [%s]' % pathNii)
        return ret
    @staticmethod
    def isGoodSeries(pseries):
        if pseries.modalityCode == 'CT' and os.path.isfile(pseries.pathNii):
            return True
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
    def __init__(self, path=None):
        if path is not None:
            self.loadInfo(path)
    def loadInfo(self, pathCase):
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
                        for idx, studyJson in enumerate(self.jsonAll['imagingStudies']):
                            dictSeries = SeriesInfo.getAllSeriesForStudy(pathCase, studyJson)
                            print ('---')


if __name__ == '__main__':
    dbdir = '../../experimental_data/dataentry_test0'
    lstCases = glob.glob('%s/case-*' % dbdir)
    numCases = len(lstCases)
    for ii,pathCase in enumerate(lstCases):
        caseInfo = CaseInfo(path=pathCase)
        print ('[%d/%d] --> case #%s' % (ii, numCases, caseInfo.caseId))