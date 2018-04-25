#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fidx = '/home/ar/github.com/BTBDB_ImageAnalysisSubPortal.git/data-real/idx-report-json.txt'
    wdir = os.path.dirname(fidx)

    dataCSV = pd.read_csv(fidx)
    pathJsons = dataCSV['path'].as_matrix()
    pathJsons = np.array([os.path.join(wdir, xx) for xx in pathJsons])

    numJsons = len(pathJsons)

    dataStatMM3 = []
    dataStatVOX = []
    for ii, ipath in enumerate(pathJsons):
        with open(ipath, 'r') as f:
            dataJson = json.loads(f.read())
            tmp = dataJson['volume']
        dataStatVOX.append([tmp['left' ][0]['value'],
                            tmp['right'][0]['value'],
                            tmp['total'][0]['value']])
        dataStatMM3.append([tmp['left' ][1]['value'],
                            tmp['right'][1]['value'],
                            tmp['total'][1]['value']])
        print ('[{}/{}] : {}'.format(ii, numJsons, os.path.basename(ipath)))
    dataStatMM3 = np.array(dataStatMM3) / (100.**3)
    dataStatVOX = np.array(dataStatVOX) / (100.**3)

    mm3_med  = np.median(dataStatMM3, axis=0)
    mm3_mean = np.mean(dataStatMM3,   axis=0)
    mm3_std  = np.std(dataStatMM3,    axis=0)

    vox_med  = np.median(dataStatVOX, axis=0)
    vox_mean = np.mean(dataStatVOX,   axis=0)
    vox_std  = np.std(dataStatVOX,    axis=0)

    dataAll  = np.hstack((dataStatMM3, dataStatVOX))
    dataStat = np.vstack((mm3_mean, mm3_med, mm3_std, vox_mean, vox_med, vox_std)).transpose().reshape((6,3))

    lstInfo = ['L(mm3)', 'R(mm3)', 'T(mm3)', 'L(vox)', 'R(vox)', 'T(vox)']

    #
    for ii in range(dataAll.shape[-1]):
        plt.subplot(2, 3, ii + 1)
        plt.grid(True)
        plt.hist(dataAll[:, ii])
        plt.title('{}*100^3, mean/med/std = {:0.2f}/{:0.2f}/{:0.2f}'
                  .format(lstInfo[ii], dataStat[ii, 0], dataStat[ii, 1], dataStat[ii, 2]))
    plt.show()

    print ('-')

