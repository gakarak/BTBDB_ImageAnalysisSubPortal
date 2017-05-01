#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import shutil
import tempfile
import subprocess
import threading
from . import checkDirContainsDicom, checkExeInPath, checkFileOrDir, mkdir_p
import mproc, log
from app.core.dataentry_v1 import DBWatcher
from datetime import datetime

#####################################
class CommandRunner(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.stdOut = None
        self.stdErr = None
        self.retCode = -1
        self.isFinished = False
    def run(self, timeOut=60):
        def target():
            self.process = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            self.stdOut, self.stdErr = self.process.communicate()
            self.retCode = self.process.returncode
            self.isFinished = True
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeOut)
        if thread.is_alive():
            self.process.terminate()
            thread.join()
            self.isFinished = True
    def checkIsOk(self, isRaiseException=True):
        ret = (self.retCode == 0) and self.isFinished
        if (not ret) and isRaiseException:
            raise Exception('Error while run [%s], stdout=[%s], stderr=[%s]' % (self.cmd, self.stdOut, self.stdErr))
        return ret

#####################################
def pydcm2nii(dirDicom, foutNii, pexe='dcm2nii'):
    # (1) check input params
    checkFileOrDir(dirDicom, isDir=True)
    checkExeInPath(pexe)
    # (2) check  dir with f*cking DICOMs
    checkDirContainsDicom(dirDicom)
    # (3) convert *.dcm --> *.nii.gz
    tmpDir = tempfile.mkdtemp(prefix='crdf-pydcm2nii-')
    runCMD = "{0} -m y -z y -r n -o {1} {2}".format(pexe, tmpDir, dirDicom)
    cmdRun1 = CommandRunner(runCMD)
    cmdRun1.run()
    cmdRun1.checkIsOk()
    #
    lstNii = sorted(glob.glob('%s/*.nii.gz' % tmpDir))
    if len(lstNii) < 1:
        raise Exception('Cant find Nifti images in dcm2nii output directory [%s]' % tmpDir)
    finpNii = lstNii[0]
    shutil.move(finpNii, foutNii)
    #
    shutil.rmtree(tmpDir)
    return os.path.isfile(foutNii)

#####################################
# Simple Runners for Task and Job conversion
class TaskRunnerConvertSeries(mproc.AbstractRunner):
    def __init__(self, series):
        self.ser = series
    def getUniqueKey(self):
        return self.ser.getKey()
    def run(self):
        # FIXME: we think, tha if series is postprocessed, then series-conversion is not needed...
        inpDirWithDicom = self.ser.getDirRaw(False)
        outPathNifti = self.ser.pathConvertedNifti(False)
        tret = pydcm2nii(dirDicom=inpDirWithDicom, foutNii=outPathNifti)
        return tret

class RunnerDBConvert(mproc.AbstractRunner):
    def __init__(self, data_dir=None):
        if data_dir is None:
            #FIXME: remove in future
            self.data_dir = 'data-cases'
        else:
            self.data_dir = data_dir
    def getUniqueKey(self):
        return 'conv-tkey-{0}'.format(datetime.now().strftime('%Y.%m.%d-%H.%M.%S:%f'))
    def run(self):
        dirData = self.data_dir
        mkdir_p(dirData)
        ptrLogger = log.get_logger(wdir=dirData, logName='s02-conv')
        dbWatcher = DBWatcher(pdir=dirData)
        ptrLogger.info(dbWatcher.toString())
        for iser, ser in enumerate(dbWatcher.allSeries()):
            if ser.isDownloaded() and (not ser.isConverted()) and (not ser.isPostprocessed()):
                convTask = TaskRunnerConvertSeries(series=ser)
                try:
                    tret = convTask.run()
                    ptrLogger.info('[%d] convert is Ok = %s, %s' % (iser, tret, ser))
                except Exception as err:
                    ptrLogger.error('[%d] Cant convert DICOM->Nifti [%s] for series %s' % (iser, err, ser))
        ptrLogger.info('Conversion is finished. Refresh DB-Info')
        dbWatcher.reload()
        ptrLogger.info(dbWatcher.toString())

#####################################
if __name__ == '__main__':
    pass