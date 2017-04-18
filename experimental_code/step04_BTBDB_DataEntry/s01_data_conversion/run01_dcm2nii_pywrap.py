#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import sys
import os
import glob
import shutil
import numpy as np
import tempfile
import distutils
import distutils.spawn
import subprocess
import threading

def checkFileOrDir(ppath, isDir=False, isRaiseException=True):
    if not isDir:
        ret = os.path.isfile(ppath)
        if isRaiseException and (not ret):
            raise Exception('Cant find file [%s]' % ppath)
    else:
        ret = os.path.isdir(ppath)
        if isRaiseException and (not ret):
            raise Exception('Cant find directory [%s]' % ppath)
    return ret

def checkExeInPath(pexe, isRaiseException=True):
    tret = distutils.spawn.find_executable(pexe)
    if isRaiseException and (tret is None):
        raise Exception('Cant find programm [%s] in {PATH}' % pexe)
    return (tret is not None)

#####################################
class CommandRunner(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.stdOut = None
        self.stdErr = None
        self.retCode = -1
        self.isFinished = False
    def run(self, timeOut=15):
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

def checkDirContainsDicom(pdir, isRaiseException=True):
    ret = '.dcm' in np.unique([os.path.splitext(os.path.join(dirWithDcm, xx))[1].lower()[:4] for xx in os.listdir(pdir) if os.path.isfile(os.path.join(dirWithDcm, xx))])
    if (not ret) and isRaiseException:
        raise Exception('Cant find DICOM files in directory [%s]' % pdir)
    return ret

def pydcm2nii(dirDicom, foutNii, pexe='dcm2nii'):
    pass

#####################################
if __name__ == '__main__':
    dirWithDcm = '../../../experimental_data/dataentry_test0/case-2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0/study-dd10657e-f2c3-48ba-87d6-b5f3fc40c752/series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2/raw'
    foutNii = 'test-out.nii.gz'
    pexe='dcm2nii'
    # (1) check input params
    checkFileOrDir(dirWithDcm, isDir=True)
    checkExeInPath(pexe)
    # (2) check  dir with f*cking DICOMs
    checkDirContainsDicom(dirWithDcm)
    # (3) convert *.dcm --> *.nii.gz
    tmpDir = tempfile.mkdtemp(prefix='crdf-pydcm2nii-')
    runCMD = "{0} -m y -z y -r n -o {1} {2}".format(pexe, tmpDir, dirWithDcm)
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