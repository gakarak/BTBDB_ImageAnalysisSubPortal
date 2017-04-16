#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import numpy as np
import scipy as sc
import nibabel as nib
import skimage.transform as sktf
from scipy.ndimage import measurements as scMeasurements
from scipy.ndimage import interpolation as scInterpolation
from scipy.ndimage import morphology as scMorphology

#############################################
# pre/postload methods FIXME: remove in feature?
def niiImagePreTransform(timg):
    return np.flip(timg.transpose((1, 0, 2)), axis=0)

def niiImagePostTransform(timg):
    return np.flip(timg, axis=0).transpose((1, 0, 2))

#############################################
# Resize 3D images
def resize3D(timg, newShape = (256, 256, 64), order=3):
    zoomScales = np.array(newShape, np.float) / np.array(timg.shape, np.float)
    ret = scInterpolation.zoom(timg, zoomScales, order=3)
    return ret

##################################
def resizeNii(pathNii, newSize=(33, 33, 33), parOrder=4, parMode='edge', parPreserveRange=True):
    if isinstance(pathNii,str) or isinstance(pathNii,unicode):
        tnii = nib.load(pathNii)
    else:
        tnii = pathNii
    timg = tnii.get_data()
    oldSize = timg.shape
    dataNew = sktf.resize(timg, newSize, order=parOrder, preserve_range=parPreserveRange, mode=parMode)
    affineOld = tnii.affine.copy()
    affineNew = tnii.affine.copy()
    k20_Old = float(oldSize[2]) / float(oldSize[0])
    k20_New = float(newSize[2]) / float(newSize[0])
    for ii in range(3):
        tCoeff = float(newSize[ii]) / float(oldSize[ii])
        if ii == 2:
            tCoeff = (affineNew[0, 0] / affineOld[0, 0]) * (k20_Old / k20_New)
        affineNew[ii, ii] *= tCoeff
        affineNew[ii,  3] *= tCoeff
    retNii = nib.Nifti1Image(dataNew, affineNew, header=tnii.header)
    return retNii

#############################################
# Simple morphology-based code for lung-mask dividing
def getCircleElement(srad=5):
    txa = np.arange(-srad, srad + 1, 1)
    tya = np.arange(-srad, srad + 1, 1)
    tza = np.arange(-srad, srad + 1, 1)
    xx, yy, zz = np.meshgrid(txa, tya, tza, sparse=True)
    F = np.sqrt(xx**2 + yy**2 + zz**2)
    ret = (F<=srad)
    return ret

def labelInfo3D(tmsk):
    timgErodeLbl, tnumLabels = sc.ndimage.label(tmsk)
    tarrSizes = []
    tarrLblId = []
    tarrCMass = []
    for ll in range(tnumLabels):
        tsiz = np.sum(timgErodeLbl==(ll+1))
        tarrSizes.append(tsiz)
        tarrLblId.append(ll+1)
        tcm = scMeasurements.center_of_mass(timgErodeLbl==(ll+1))
        tarrCMass.append(tcm)
    tarrCMass  = np.array(tarrCMass)
    tarrLblId  = np.array(tarrLblId)
    tarrSizes  = np.array(tarrSizes, np.float)
    tarrSizes /= np.sum(tarrSizes)
    if tnumLabels>1:
        idxSort = np.argsort(-tarrSizes)
        tarrSizes = tarrSizes[idxSort]
        tarrLblId = tarrLblId[idxSort]
        tarrCMass = tarrCMass[idxSort,:]
    return (timgErodeLbl, tarrSizes, tarrLblId, tarrCMass, tnumLabels)

def morph3DIter(timg, pElem, numIter, isErosion=True):
    ret = timg
    for ii in range(numIter):
        if isErosion:
            ret = scMorphology.binary_erosion(ret, pElem)
        else:
            ret = scMorphology.binary_dilation(ret, pElem)
    return ret

def makeLungedMaskNii(pimgNii, parStrElemSize=2, parNumIterMax=9, isDebug=False):
    if isinstance(pimgNii, str) or isinstance(pimgNii, unicode):
        pimgNii = nib.load(pimgNii)
    timg = niiImagePreTransform(pimgNii.get_data())
    retMsk, retIsOk = makeLungedMask(timg,
                                          parStrElemSize=parStrElemSize,
                                          parNumIterMax=parNumIterMax,
                                          isDebug=isDebug)
    retMskNii = nib.Nifti1Image(niiImagePostTransform(retMsk).astype(pimgNii.get_data_dtype()),
                                pimgNii.affine,
                                header=pimgNii.header)
    return (retMskNii, retIsOk)

def makeLungedMask(timg, parStrElemSize=2, parNumIterMax=9, isDebug=False):
    strSiz = parStrElemSize
    strElem = getCircleElement(strSiz)
    # strElem = scMorphology.generate_binary_structure(3,3)
    numIterMax = parNumIterMax
    timgErode = timg.copy()
    numIterReal = 0
    timgDiv = None
    for it in range(numIterMax):
        timgErode = scMorphology.binary_erosion(timgErode, strElem)
        numIterReal += 1
        timgErodeLbl, tarrSizesSorted, tarrLblIdSorted, retCM, tnumLabels = labelInfo3D(timgErode)
        if isDebug:
            print ('[%d/%d] : %s' % (it, numIterMax, tarrSizesSorted.tolist()))
        if tnumLabels > 2:
            tmin2 = np.min(tarrSizesSorted[1:2])
            if tmin2 > 0.15:
                timgDiv = np.zeros(timg.shape)
                retCM_X = retCM[:2, 1]
                idxSortX = np.argsort(retCM_X)
                for lli in range(2):
                    timgErodedL = (timgErodeLbl == tarrLblIdSorted[idxSortX[lli]])
                    timgDilateL = morph3DIter(timgErodedL, pElem=strElem, numIter=(numIterReal+0), isErosion=False)
                    tlungMsk = ((timg > 0) & timgDilateL)
                    # (1) Fill holes:
                    tlungMsk = scMorphology.binary_fill_holes(tlungMsk)
                    # (2) Last filter potential small regions
                    retLbl, retSizSrt, retLblSrt, _, _ = labelInfo3D(tlungMsk)
                    tlungMsk = (retLbl == retLblSrt[0])
                    timgDiv[tlungMsk] = (idxSortX[lli] + 1)
                if isDebug:
                    print (':: isOk, #Iter = %d' % numIterReal)
                break
            else:
                tfirstLbl = tarrLblIdSorted[0]
                timgErode = (timgErodeLbl == tfirstLbl)
        else:
            timgErode = (timgErodeLbl>0)
    # Final processing: if we did not find two separate lung volumes - serach biggest component and filter other data
    retMsk = np.zeros(timg.shape)
    if timgDiv is None:
        retLbl, retSizSrt, retLblSrt, retCM, _ = labelInfo3D(timg > 0)
        retCM_X = retCM[:2, 1]
        if retCM_X[retLblSrt[0]-1]<(timg.shape[1]/2):
            # Left lung
            retMsk[retLbl == retLblSrt[0]] = 1
        else:
            # Right lung
            retMsk[retLbl == retLblSrt[0]] = 2
        # retMsk[retLbl == retLblSrt[0]] = 3  # 1st component
        otherData = ((~(retMsk>0)) & (timg > 0))
        retMsk[otherData] = 4  # other data
        isOk = False
    else:
        retMsk = timgDiv
        isOk = True
    return (retMsk, isOk)

#############################################
# Basic Lung-information extraction code
def prepareLungSizeInfoNii(mskLungsNii, isInStrings=False, isPreproc=True):
    if isPreproc:
        timg = niiImagePreTransform(mskLungsNii.get_data())
    else:
        timg = mskLungsNii.get_data()
    return prepareLungSizeInfo(mskLungs=timg,
                               niiHeader=mskLungsNii.header,
                               isInStrings=isInStrings)

def prepareLungSizeInfo(mskLungs, niiHeader, isInStrings=False):
    isOk = bool(np.unique(mskLungs).max()<3)
    hdrXyzUnits = niiHeader.get_xyzt_units()
    if len(hdrXyzUnits) > 1:
        hdrXyzUnits = hdrXyzUnits[0]
    else:
        hdrXyzUnits = 'unknown'
    hdrVoxelSize = float(np.prod(niiHeader.get_zooms()))
    if isOk:
        volLungLeft  = int(np.sum(mskLungs == 1))
        volLungRight = int(np.sum(mskLungs == 2))
        volLungTotal = volLungLeft + volLungRight
    else:
        volLungLeft  = -1
        volLungRight = -1
        volLungTotal = int(np.sum(mskLungs == 3))
    sizeLungLeft  = volLungLeft  * hdrVoxelSize
    sizeLungRight = volLungRight * hdrVoxelSize
    sizeLungTotal = volLungTotal * hdrVoxelSize
    if isInStrings:
        volLungLeft   = '%0.1f' % volLungLeft
        volLungRight  = '%0.1f' % volLungRight
        volLungTotal  = '%0.1f' % volLungTotal
        sizeLungLeft  = '%0.1f' % sizeLungLeft
        sizeLungRight = '%0.1f' % sizeLungRight
        sizeLungTotal = '%0.1f' % sizeLungTotal
        hdrVoxelSize  = '%0.3f' % hdrVoxelSize
    retInfoLungSizes = {
        'units':         hdrXyzUnits,
        'voxelSize':     hdrVoxelSize,
        'isValidLungs':  isOk,
        'lungsVoxels': {
            'left':     volLungLeft,
            'right':    volLungRight,
            'total':    volLungTotal,
        },
        'lungsSizes': {
            'left':     sizeLungLeft,
            'right':    sizeLungRight,
            'total':    sizeLungTotal
        },
    }
    return retInfoLungSizes

if __name__ == '__main__':
    pass