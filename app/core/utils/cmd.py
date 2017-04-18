#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import shutil
import tempfile
import subprocess
import threading
from . import checkDirContainsDicom, checkExeInPath, checkFileOrDir

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
if __name__ == '__main__':
    pass