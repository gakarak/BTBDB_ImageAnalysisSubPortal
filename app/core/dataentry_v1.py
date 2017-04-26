#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import json
import numpy as np
import nibabel as nib
import utils

#######################################
class SeriesInfo:
    # (1) Study:
    studyId = None
    # (2) Series:
    ptrCase  = None
    jsonInfo = None
    def __init__(self, ptrCase=None, studyId=None, jsonInfo=None):
        self.ptrCase = ptrCase
        self.studyId = studyId
        self.jsonInfo = jsonInfo
    def toString(self):
        return 'Series: [{0}]'.format(self.getDir())
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def isInitialized(self):
        if (self.ptrCase is not None) and (self.studyId is not None) and (self.jsonInfo is not None):
            return True
        return False
    def uid(self):
        if self.isInitialized():
            return self.jsonInfo['uid']
    def getDirRaw(self, isRelative=True):
        tmp = self.getDir(isRelative=isRelative)
        if tmp is not None:
            return os.path.join(tmp, 'raw')
    def getDir(self, isRelative=True):
        if self.isInitialized():
            caseId = self.ptrCase.caseId()
            pdir = None
            if not isRelative:
                pdir = self.ptrCase.wdir()
            return CaseInfo.getRelativeSeriesPath(caseId=caseId, studyId=self.studyId,
                                                  seriesUid=self.uid(), modality=self.modality(), dataDir=pdir)
    def caseId(self):
        if self.ptrCase is not None:
            return self.ptrCase.caseId()
    def studyUID(self):
        if self.ptrCase is not None:
            return self.ptrCase.getStudyUidById(self.studyId)
    def modality(self):
        if self.jsonInfo is not None:
            return self.jsonInfo['modality']['code']
    def numInstances(self):
        if self.jsonInfo is not None:
            return self.jsonInfo['numberOfInstances']
        else:
            return 0
    def isGood(self):
        ret = False
        if (self.ptrCase is not None) and (self.jsonInfo is not None):
            mod = self.modality()
            num = self.numInstances()
            if (mod in CaseInfo.GOOD_MODALITIES.keys()) and (num>=CaseInfo.GOOD_MODALITIES[mod]):
                ret = True
        return ret
    def getKey(self):
        if self.isInitialized():
            return '%s:%s' % (self.studyId, self.uid())
        else:
            return 'invalid-key'
    # *** Helpers *** : series state cheking
    # [Check #1] : check download state
    def getInstancesJs(self):
        if self.isInitialized():
            return self.jsonInfo['instance']
        else:
            return []
    def getPathInstances(self, isRelative = True):
        ret = []
        if self.isInitialized():
            dirRaw = self.getDirRaw(isRelative = isRelative)
            instancesJs = self.getInstancesJs()
            for instJs in instancesJs:
                tpath = os.path.join(dirRaw, self.getInstanceBaseNameJs(instJs))
                ret.append(tpath)
            return ret
        return ret
    def isHasData(self):
        return (len(self.getInstancesJs())>0)
    def isDownloaded(self):
        sdir = self.getDir(isRelative=False)
        if not os.path.isdir(sdir):
            return False
        pathsInstances = self.getPathInstances(isRelative=False)
        #FIXME: simplified version: downloaded file is not validated (by size for example)
        for path in pathsInstances:
            if not os.path.isfile(path):
                return False
        return True
    # [Check #2] : check data conversion status
    def pathConvertedNifti(self, isRelative=True):
        if self.isInitialized():
            return '{0}.nii.gz'.format(self.getDir(isRelative=isRelative))
    def isConverted(self):
        pathNii = self.pathConvertedNifti(isRelative=False)
        if pathNii is not None:
            return os.path.isfile(pathNii)
        return False
    # [Check #3] : check that data os postprocessed (lung segmented, lesion deted, report (json and PDF) is prepared)
    def pathPostprocLungs(self, isRelative=True):
        if self.isInitialized():
            return '{0}-lungs.nii.gz'.format(self.getDir(isRelative=isRelative))
    def pathPostprocLesions(self, isRelative=True):
        if self.isInitialized():
            return '{0}-lesions.nii.gz'.format(self.getDir(isRelative=isRelative))
    def pathPostprocPreview(self, isRelative=True, previewId=0):
        if self.isInitialized():
            return '{0}-preview{1}.jpg'.format(self.getDir(isRelative=isRelative), previewId)
    def pathPostprocReport(self, isRelative=True):
        if self.isInitialized():
            return '{0}-report.json'.format(self.getDir(isRelative=isRelative))
    def pathPostprocReportPDF(self, isRelative=True):
        if self.isInitialized():
            return '{0}-report.pdf'.format(self.getDir(isRelative=isRelative))
    def isPostprocessed(self):
        if self.isInitialized():
            pathLungs     = self.pathPostprocLungs(isRelative=False)
            pathLesion    = self.pathPostprocLesions(isRelative=False)
            pathReport    = self.pathPostprocReport(isRelative=False)
            pathReportPDF = self.pathPostprocReportPDF(isRelative=False)
            if not os.path.isfile(pathLungs):
                return False
            if not os.path.isfile(pathLesion):
                return False
            if not os.path.isfile(pathReport):
                return False
            # if not os.path.isfile(pathReportPDF):
            #     return False
            return True
        return False
    def getReportJson(self, root_url=None):
        pathReport = self.pathPostprocReport(isRelative=False)
        with open(pathReport, 'r') as f:
            jsonData = json.loads(f.read())
            dir_relative = os.path.dirname(self.getDir(isRelative=True))
            if jsonData.has_key('preview_images'):
                for tmp in jsonData['preview_images']:
                    if root_url is None:
                        tmp['url'] = '/{0}/{1}'.format(dir_relative, os.path.basename(tmp['url']))
                    else:
                        tmp['url'] = '{0}{1}/{2}'.format(root_url, dir_relative, os.path.basename(tmp['url']))
            return jsonData
    @classmethod
    def getInstanceBaseName(cls, instanceId, instanceNum):
        return '{0:04d}-{1}.dcm'.format(instanceNum, instanceId)
    @classmethod
    def getInstanceBaseNameJs(cls, instanceJs):
        instanceId  = instanceJs['uid']
        instanceNum = instanceJs['number']
        return cls.getInstanceBaseName(instanceId, instanceNum)
    @staticmethod
    def getAllSeriesForStudy(caseInfo, studyJson, isDropBad = False):
        studyID = studyJson['id']
        # studyUID = studyJson['studyUid']
        # studyDate = studyJson['studyDate']
        ret = dict()
        if 'series' in studyJson.keys():
            for ssi, seriesJson in enumerate(studyJson['series']):
                pser = SeriesInfo(ptrCase=caseInfo, studyId=studyID, jsonInfo=seriesJson)
                skey = pser.getKey()
                if (not isDropBad) or pser.isGood():
                    ret[skey] = pser
        return ret
    @staticmethod
    def isGoodSeries(pseries):
        return pseries.isGood()

#######################################
class CaseInfo:
    GOOD_MODALITIES = {'CT': 36}
    # GOOD_MODALITIES = {'CT': 36, 'XR': 1, 'CR': 1}
    #
    JSON_SHORT = 'info-short.json'
    JSON_ALL = 'info-all.json'
    dictShort = None
    dictAll = None
    path = None
    series = None
    #
    def isInitialized(self):
        return (self.dictShort is not None) and (self.dictAll is not None)
    def seriesNum(self):
        if self.series is not None:
            return len(self.series)
        else:
            return 0
    def wdir(self):
        if self.path is not None:
            return os.path.dirname(self.path)
        else:
            return None
    def caseId(self):
        if self.dictShort is not None:
            return self.dictShort['conditionId']
        else:
            return None
    def patientId(self):
        if self.dictShort is not None:
            return self.dictShort['patientId']
        else:
            return None
    def isEmpty(self):
        if self.series is not None:
            return (len(self.series)<1)
        else:
            return True
    def toString(self):
        tmp='CaseInfo: caseId={0}, patientId={1}, #series={2}'.format(self.caseId(), self.patientId(), self.seriesNum())
        return tmp
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def __init__(self, path=None):
        if path is not None:
            self.loadInfo(path)
    def getKey(self):
        # return '%s:%s' % (self.caseId(), self.patientId())
        return self.caseId()
    def getStudyUidById(self, studyID):
        if self.isInitialized() and (self.dictAll['imagingStudies'] is not None):
            for study in self.dictAll['imagingStudies']:
                if studyID == study['id']:
                    return study['studyUid']
    def loadInfoById(self, dataDir, caseId, isDropBad = False):
        case_dir = CaseInfo.getCaseDirName(caseId, dataDir=dataDir)
        isOk = False
        if os.path.isdir(case_dir):
            self.loadInfo(case_dir, isDropBad=isDropBad)
            isOk = True
        return isOk
    def setupFromJson(self, jsonShort, jsonAll):
        self.dictShort = jsonShort
        self.dictAll = jsonAll
    def loadSeriesInfo(self, isDropBad=False):
        if (self.dictAll is not None) and ('imagingStudies' in self.dictAll.keys()):
            if self.dictAll['imagingStudies'] is not None:
                tdict = dict()
                for idx, studyJson in enumerate(self.dictAll['imagingStudies']):
                    dictSeries = SeriesInfo.getAllSeriesForStudy(self, studyJson, isDropBad=isDropBad)
                    tdict = dict(tdict.items() + dictSeries.items())
                    # print ('Case [%s] : #Series = %d' % (self.caseId, len(self.series)))
                self.series = tdict
    def loadInfo(self, pathCase, isDropBad = False):
        self.path = pathCase
        # self.wdir = os.path.dirname(pathCase)
        tpathJsonShort = os.path.join(pathCase, self.JSON_SHORT)
        tpathJsonAll = os.path.join(pathCase, self.JSON_ALL)
        if os.path.isfile(tpathJsonShort):
            with open(tpathJsonShort, 'r') as f:
                self.dictShort = json.load(f)
        if os.path.isfile(tpathJsonAll):
            with open(tpathJsonAll, 'r') as f:
                self.dictAll = json.load(f)
                if 'imagingStudies' in self.dictAll.keys():
                    self.loadSeriesInfo(isDropBad=isDropBad)
    def save2Disk(self):
        if (self.wdir() is not None) and (self.dictAll is not None) and (self.dictShort is not None):
            case_dir = '%s/%s' % (self.wdir(), CaseInfo.getCaseDirName(self.caseId()))
            utils.mkdir_p(case_dir)
            fjsonAll  = os.path.join(case_dir, CaseInfo.JSON_ALL)
            fjsonShort = os.path.join(case_dir, CaseInfo.JSON_SHORT)
            with open(fjsonAll, 'w') as fall, open(fjsonShort, 'w') as fshort:
                fall.write(json.dumps(self.dictAll, indent=4))
                fshort.write(json.dumps(self.dictShort, indent=4))
    def getGoodSeries(self, cfg=None):
        ret = []
        if self.isInitialized():
            if ('imagingStudies' in self.dictAll) and (self.dictAll['imagingStudies'] is not None):
                for iStudyJs, studyJs in enumerate(self.dictAll['imagingStudies']):
                    studyId = studyJs['id']
                    for iSeriesJs, seriesJs in enumerate(studyJs['series']):
                        numInstances = seriesJs['numberOfInstances']
                        mod = seriesJs['modality']['code']
                        if (mod in CaseInfo.GOOD_MODALITIES.keys()) and (numInstances>=CaseInfo.GOOD_MODALITIES[mod]):
                            series = SeriesInfo(ptrCase=self, studyId=studyId, jsonInfo=seriesJs)
                            ret.append(series)
        return ret
    @staticmethod
    def getCaseDirName(caseId, dataDir=None):
        if dataDir is None:
            return 'case-{0}'.format(caseId)
        else:
            return '{1}/case-{0}'.format(caseId, dataDir)
    @staticmethod
    def getCaseDirNameJs(jsonAll, dataDir=None):
        if dataDir is None:
            return 'case-{0}'.format(jsonAll['id'])
        else:
            return '{1}/case-{0}'.format(jsonAll['id'], dataDir)
    @staticmethod
    def getStudyDirName(studyId):
        return 'study-{0}'.format(studyId)
    @staticmethod
    def getStudyDirNameJs(studyJs):
        return 'study-{0}'.format(studyJs['id'])
    @staticmethod
    def getSeriesDirName(seriesUid, modality):
        return 'series-{0}-{1}'.format(seriesUid, modality)
    @staticmethod
    def getSeriesDirNameJs(seriesJs):
        seriesUid = seriesJs['uid']
        modality = seriesJs['uid']['modality']['code']
        return 'series-{0}-{1}'.format(seriesUid, modality)
    @staticmethod
    def getRelativeSeriesPath(caseId, studyId, seriesUid, modality, dataDir = None):
        tmp = '{0}/{1}/{2}'.format(
            CaseInfo.getCaseDirName(caseId),
            CaseInfo.getStudyDirName(studyId),
            CaseInfo.getSeriesDirName(seriesUid, modality))
        if dataDir is None:
            return tmp
        else:
            return os.path.join(dataDir, tmp)
    @staticmethod
    def getRelativeSeriesPathJs(jsonAll, jsonStudy, jsonSeries, dataDir = None):
        tmp = '{0}/{1}/{2}'.format(
            CaseInfo.getCaseDirNameJs(jsonAll),
            CaseInfo.getStudyDirNameJs(jsonStudy),
            CaseInfo.getSeriesDirNameJs(jsonSeries))
        if dataDir is None:
            return tmp
        else:
            return os.path.join(dataDir, tmp)
    @staticmethod
    def newCase(dataDir, dictShort, dictAll, isSaveToDisk=False):
        case = CaseInfo()
        case.path = CaseInfo.getCaseDirNameJs(dictAll, dataDir=dataDir)
        case.setupFromJson(jsonShort=dictShort, jsonAll=dictAll)
        if isSaveToDisk:
            case.save2Disk()
        return case

#######################################
class DBWatcher:
    wdir = None
    cases = None
    def __init__(self, pdir = None):
        self.cases = dict()
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
                print ('[%d/%d] --> case #%s' % (ii, numCases, caseInfo.caseId()))
            if isDropEmpty and caseInfo.isEmpty():
                continue
            else:
                tkey = caseInfo.getKey()
                dictCases[tkey] = caseInfo
        self.cases = dictCases
    def reload(self):
        if self.wdir is not None:
            self.load(self.wdir)
    def isHaveCase(self, caseId):
        if self.cases is None:
            return False
        else:
            if caseId in self.cases.keys():
                return True
            else:
                return False
    def allSeries(self):
        for kcase, case in self.cases.items():
            if not case.isEmpty():
                for _, ser in case.series.items():
                    yield ser
    def checkSeriesInDB(self, pseries):
        for series in self.allSeries():
            if series.getKey() == pseries.getKey():
                return True
        return False
    def toString(self):
        if self.cases is None:
            return 'DBWatcher is not initialized'
        else:
            tret = self.getStatistics()
            return 'Cases: #All/#Good = {0}/{1}, Series = #All/#Good/#Down/#Conv/#Proc = {2}/{3}/{4}/{5}/{6}'\
                .format(tret['cases']['total'],
                        tret['cases']['empty'],
                        tret['series']['total'],
                        tret['series']['good'],
                        tret['series']['downloaded'],
                        tret['series']['converted'],
                        tret['series']['processed'])
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def getStatistics(self):
        ret = {
            'cases': {
                'total': 0,
                'empty': 0
            },
            'series': {
                'total': 0,
                'good':  0,
                'downloaded': 0,
                'converted': 0,
                'processed': 0
            }
        }
        if self.cases is not None:
            ret['cases']['total'] = len(self.cases)
            for ic, (kcase, case) in enumerate(self.cases.items()):
                if case.isEmpty():
                    ret['cases']['empty'] += 1
                else:
                    for si, (kstudy, ser) in enumerate(case.series.items()):
                        ret['series']['total'] += 1
                        if ser.isGood():
                            ret['series']['good'] += 1
                        if ser.isDownloaded():
                            ret['series']['downloaded'] += 1
                        if ser.isConverted():
                            ret['series']['converted'] += 1
                        if ser.isPostprocessed():
                            ret['series']['processed'] += 1
        return ret
    def printStat(self):
        print (self.toString())

#######################################
if __name__ == '__main__':
    pass