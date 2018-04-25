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

    lstInfo = ['L(mm3)', 'R(mm3)', 'T(mm3)', 'L(vox)', 'R(vox)', 'T(vox)']
    lstData = []

    #
    plt.subplot(2, 3, 1)
    plt.hist(dataStatMM3[:, 0])
    plt.title('L(mm3), mean/median/std = {:0.2f}/{:0.2f}/{:0.2f}'.format(mm3_mean[0], mm3_med[0], mm3_std[0]))
    plt.grid(True)
    plt.subplot(2, 3, 2)
    plt.hist(dataStatMM3[:, 1])
    plt.title('R(mm3), mean/median/std = {:0.2f}/{:0.2f}/{:0.2f}'.format(mm3_mean[1], mm3_med[1], mm3_std[1]))
    plt.grid(True)
    plt.subplot(2, 3, 3)
    plt.hist(dataStatMM3[:, 2])
    plt.title('T(mm3), mean/median/std = {:0.2f}/{:0.2f}/{:0.2f}'.format(mm3_mean[2], mm3_med[2], mm3_std[2]))
    plt.grid(True)
    #
    plt.subplot(2, 3, 4)
    plt.hist(dataStatVOX[:, 0])
    plt.title('L(vox), mean/median/std = {:0.2f}/{:0.2f}/{:0.2f}'.format(vox_mean[0], vox_med[0], vox_std[0]))
    plt.grid(True)
    plt.subplot(2, 3, 5)
    plt.hist(dataStatVOX[:, 1])
    plt.title('R(vox), mean/median/std = {:0.2f}/{:0.2f}/{:0.2f}'.format(vox_mean[1], vox_med[1], vox_std[1]))
    plt.grid(True)
    plt.subplot(2, 3, 6)
    plt.hist(dataStatVOX[:, 2])
    plt.title('T(vox), mean/median/std = {:0.2f}/{:0.2f}/{:0.2f}'.format(vox_mean[2], vox_med[2], vox_std[2]))
    plt.grid(True)
    plt.show()

    print ('-')

