#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

from app.core.dataentry_v1 import DBWatcher

if __name__ == '__main__':
    dataDir = '../s02_mproc/data-cases'
    dbWatcher = DBWatcher(pdir=dataDir)
    print (dbWatcher)
    responseCases = []
    for kcase, case in dbWatcher.cases.items():
        retAge = case.dictShort['ageOnset']
        retDiag = case.dictShort['diagnosis']['display']
        retGender = case.dictShort['patientGender']
        arrSeries = []
        if case.isInitialized() and (not case.isEmpty()):
            for kseries, series in case.series.items():
                if series.isPostprocessed():
                    retSeries = {
                        'case_id': case.caseId(),
                        'patient_id': case.patientId(),
                        'study_uid': series.studyUID(),
                        'series_uid': series.uid()
                    }
                    arrSeries.append(retSeries)
        if len(arrSeries)>0:
            retCase = {
                'age': retAge,
                'diag': retDiag,
                'gender': retGender,
                'series': arrSeries
            }
            responseCases.append(retCase)
    print ('---')
