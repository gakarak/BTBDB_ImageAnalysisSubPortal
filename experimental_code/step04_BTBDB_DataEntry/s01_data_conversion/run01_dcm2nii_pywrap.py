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
from subprocess import Popen, PIPE
import shlex

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
            self.process = subprocess.Popen(self.cmd, stdout=PIPE, stderr=PIPE, shell=True)
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
if __name__ == '__main__':
    dirWithDcm = '../../../experimental_data/dataentry_test0/case-2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0/study-dd10657e-f2c3-48ba-87d6-b5f3fc40c752/series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2/raw'
    foutNii = 'test-out.nii.gz'

    pexe='dcm2nii'
    #
    checkFileOrDir(dirWithDcm, isDir=True)
    checkExeInPath(pexe)
    #
    tmpDir = tempfile.mkdtemp(prefix='crdf-pydcm2nii-')
    runCMD = "{0} -m y -z y -r n -o {1} {2}".format(pexe, tmpDir, dirWithDcm)
    cmdRun1 = CommandRunner(runCMD)
    cmdRun1.run()
    cmdRun1.checkIsOk()
    #

    #
    shutil.rmtree(tmpDir)
    print ('---')