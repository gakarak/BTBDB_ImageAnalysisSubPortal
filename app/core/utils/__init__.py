#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os

import numpy as np
import tempfile
import distutils
import distutils.spawn
import errno

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

if __name__ == '__main__':
    pass