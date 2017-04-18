#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os

import numpy as np
import tempfile
import distutils
import distutils.spawn

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

def checkDirContainsDicom(pdir, isRaiseException=True):
    ret = '.dcm' in np.unique([os.path.splitext(os.path.join(pdir, xx))[1].lower()[:4] for xx in os.listdir(pdir) if os.path.isfile(os.path.join(pdir, xx))])
    if (not ret) and isRaiseException:
        raise Exception('Cant find DICOM files in directory [%s]' % pdir)
    return ret

if __name__ == '__main__':
    pass