#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import app.core.preprocessing as preproc

###############################
def normalizeCTImage(pimg, outType = np.uint8):
    pimg = pimg.astype(np.float)
    vMin = -1000.
    vMax = +200.
    ret = 255. * (pimg - vMin) / (vMax - vMin)
    ret[ret < 0] = 0
    ret[ret > 255] = 255.
    return ret.astype(outType)

def makePreview4Lesion(dataImg, dataMsk, dataLes, sizPrv=256, nx=4, ny=3, pad=5, lesT=0.7):
    shpPrv = (sizPrv, sizPrv, sizPrv)
    dataImgR = preproc.resize3D(dataImg, shpPrv)
    dataMskR = preproc.resize3D(dataMsk, shpPrv)
    dataLesR = preproc.resize3D(dataLes, shpPrv)
    numXY = nx*ny - 1
    brd = 0.1
    arrZ = np.linspace(brd*sizPrv, (1.-brd)*sizPrv, numXY ).astype(np.int)
    cnt = 0
    tmpV = []
    for yy in range(ny):
        tmpH = []
        for xx in range(nx):
            if (yy==0) and (xx==0):
                timg = np.rot90(dataImgR[:, sizPrv / 2, :])
                tmsk = np.rot90(dataMskR[:, sizPrv / 2, :] > 0.1)
                tles0 = np.rot90(dataLesR[:, sizPrv / 2, :])
            else:
                zidx = arrZ[cnt]
                timg = np.rot90(dataImgR[:, :, zidx])
                tmsk = np.rot90(dataMskR[:, :, zidx] > 0.1)
                tles0 = np.rot90(dataLesR[:, :, zidx])
                cnt+=1
            timg = timg.astype(np.float)
            timgR = timg.copy()
            timgG = timg.copy()
            timgB = timg.copy()
            timgR[tmsk>0] += 30
            timgG[tmsk>0] += 30
            #
            tles = tles0.copy()
            tles[tmsk<1] = 0
            tlesT0 = 0.1
            tlesT1 = lesT
            tles /= tlesT1
            tles[tles>1.] = 1.
            tles[tles<tlesT0] = 0
            tles = (255.*tles).astype(np.uint8)
            timgR[tles>1] = tles[tles>1]
            timgRGB = np.dstack([timgR, timgG, timgB])
            timgRGB[timgRGB>255] = 255
            timgRGB = timgRGB.astype(np.uint8)
            timgRGB = np.pad(timgRGB, pad_width=[[pad],[pad],[0]], mode='constant')
            tmpH.append(timgRGB)
        imgH = np.hstack(tmpH)
        tmpV.append(imgH)
    imgV = np.vstack(tmpV)
    return imgV

###############################
if __name__ == '__main__':
    fnii = '../../experimental_data/dataentry_test0/case-2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0/study-dd10657e-f2c3-48ba-87d6-b5f3fc40c752/series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2-CT.nii.gz'
    fniiMsk = '%s-lungs.nii.gz' % fnii
    fniiLes = '%s-lesion.nii.gz' % fnii
    #
    dataImg = normalizeCTImage(nib.load(fnii).get_data())
    dataMsk = nib.load(fniiMsk).get_data()
    dataLes = nib.load(fniiLes).get_data()
    #
    imgPreview = makePreview4Lesion(dataImg, dataMsk, dataLes)
    plt.imshow(imgPreview)
    plt.show()
    print('--')
