#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import matplotlib.pyplot as plt
import nibabel as nib
try:
    import common as comm
except:
    from . import common as comm
import skimage.io as skio
import skimage.transform as sktf

#################################
if __name__ == '__main__':
    dbdir = '../../experimental_data/dataentry_test0'
    dbWatcher = comm.DBWatcher()
    dbWatcher.load(dbdir, isDropEmpty=True, isDropBadSeries=True)
    dbWatcher.printStat()
    for ii, ser in enumerate(dbWatcher.allSeries()):
        print ('%d [%s] : %s' % (ii, ser.getKey(), ser.shape))
        # dataNii = nib.load(ser.pathNii).get_data()
        # print ('---')