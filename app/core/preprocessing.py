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
from app.core.segmct.fcnn_lesion3dv2 import get_overlay_msk, lesion_id2name, lesion_name2id, lesion_id2rgb
import skimage as sk
import skimage.filters
import skimage.morphology
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from copy import deepcopy
import os
import json
import shutil
import SimpleITK as sitk
import time

#############################################
def _get_msk_bnd2(pmsk, dilat_siz=2):
    ret_msk_bnd = skimage.filters.laplace(pmsk)>0
    if dilat_siz is not None:
        tsqr = sk.morphology.square(dilat_siz)
        ret_msk_bnd = sk.morphology.binary_dilation(ret_msk_bnd, tsqr)
    return ret_msk_bnd

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
    ret = scInterpolation.zoom(timg, zoomScales, order=order)
    return ret

##################################
def resizeNii(pathNii, newSize=(33, 33, 33), parOrder=4, parMode='edge', parPreserveRange=True):
    if isinstance(pathNii,str):# or isinstance(pathNii,unicode):
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
    if isinstance(pimgNii, str):# or isinstance(pimgNii, unicode):
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
    # (1) get spacing
    tmp_spacing = niiHeader.get_zooms()
    if (tmp_spacing is not None) and (len(tmp_spacing)==3):
        ret_spacing = {
            'x': float(tmp_spacing[0]),
            'y': float(tmp_spacing[1]),
            'z': float(tmp_spacing[2])
        }
    else:
        ret_spacing = {
            'x': -1,
            'y': -1,
            'z': -1
        }
    # (2) get bounding_box
    tmp_bounding_box =  niiHeader.get_data_shape()
    if (tmp_bounding_box is not None) and (len(tmp_bounding_box) == 3):
        bounding_box_vox = {
            'units': 'voxels',
            'x': tmp_bounding_box[0],
            'y': tmp_bounding_box[1],
            'z': tmp_bounding_box[2],
        }
        bounding_box_mm = {
            'units': 'mm',
            'x': tmp_bounding_box[0] * ret_spacing['x'],
            'y': tmp_bounding_box[1] * ret_spacing['y'],
            'z': tmp_bounding_box[2] * ret_spacing['z'],
        }
    else:
        bounding_box_vox = {
            'units': 'voxels',
            'x': -1,
            'y': -1,
            'z': -1,
        }
        bounding_box_mm = {
            'units': 'mm',
            'x': -1,
            'y': -1,
            'z': -1,
        }
    bounding_box = [
        bounding_box_vox,
        bounding_box_mm
    ]
    # (3) calc assimetry
    try:
        assimetryVol = np.abs( float(sizeLungLeft) - float(sizeLungRight)) / (float(sizeLungLeft) + float(sizeLungRight))
    except:
        assimetryVol = -1
    assimatryTex = -1
    # number -> strings
    if isInStrings:
        volLungLeft   = '%0.1f' % volLungLeft
        volLungRight  = '%0.1f' % volLungRight
        volLungTotal  = '%0.1f' % volLungTotal
        sizeLungLeft  = '%0.1f' % sizeLungLeft
        sizeLungRight = '%0.1f' % sizeLungRight
        sizeLungTotal = '%0.1f' % sizeLungTotal
        hdrVoxelSize  = '%0.3f' % hdrVoxelSize
    # Fin
    retInfoLungSizes = {
        'units':         hdrXyzUnits,
        'voxelSize':     hdrVoxelSize,
        'isValidLungs':  isOk,
        'number_of_slices': bounding_box_vox['z'],
        'spacing': ret_spacing,
        'bounding_box': bounding_box,
        'volume': {
            'total': [
                {
                    'units': 'voxels',
                    'value': volLungTotal
                },
                {
                    'units': 'mm3',
                    'value': sizeLungTotal
                },
            ],
            'left': [
                {
                    'units': 'voxels',
                    'value': volLungLeft
                },
                {
                    'units': 'mm3',
                    'value': sizeLungLeft
                }
            ],
            'right': [
                {
                    'units': 'voxels',
                    'value': volLungRight
                },
                {
                    'units': 'mm3',
                    'value': sizeLungRight
                }
            ]
        },
        'asymmetry': [
            {
                "type": "volume",
                "value": assimetryVol
            },
            {
                "type": "texture",
                "value": assimatryTex
            }
        ],
        # 'lungsVoxels': {
        #     'left':     volLungLeft,
        #     'right':    volLungRight,
        #     'total':    volLungTotal,
        # },
        # 'lungsSizes': {
        #     'left':     sizeLungLeft,
        #     'right':    sizeLungRight,
        #     'total':    sizeLungTotal
        # },
    }
    return retInfoLungSizes

def getJsonReport(series, reportLesionScore, reportLungs, reportLesion = None, lstImgJson=[], reportLesionScoreById = None, reportLesionScoreByName = None):
    case_id = series.ptrCase.caseId()
    patient_id = series.ptrCase.patientId()
    study_uid = series.studyUID()
    study_id  = series.studyId
    series_uid = series.uid()
    retLesions = {}
    # retLesions = {
    #     'left': None,
    #     'right': None
    # }
    if reportLesion is None:
        k2n = {1: 'left', 2: 'right'}
        for k, v in reportLesionScore.items():
            k_name = k2n[k]
            retLesions[k_name] = v
            if reportLesionScoreById is not None:
                retLesions[k_name + '_by_id'] = reportLesionScoreById[k]
            if reportLesionScoreByName is not None:
                retLesions[k_name + '_by_name'] = reportLesionScoreByName[k]
            # if k == 1:
            #     retLesions['left'] = v
            #
            #     retLesions['left_by_id'] =
            # if k == 2:
            #     retLesions['right'] = v
    else:
        retLesions = reportLesion['lesions']
    ret = {
        'case_id' : case_id,
        'patient_id' : patient_id,
        'study_uid' : study_uid,
        'study_id': study_id,
        'series_uid' : series_uid,
        'number_of_slices' : reportLungs['number_of_slices'],
        'spacing' : reportLungs['spacing'],
        'bounding_box': reportLungs['bounding_box'],
        'volume': reportLungs['volume'],
        'asymmetry': reportLungs['asymmetry'],
        'preview_images' : lstImgJson,
        'lesions': retLesions,
        'comment': 'No comments...'
    }
    return ret

#############################################
def getMinMaxLungZ(pmsk):
    zsum = np.sum(pmsk, axis=(0, 1))
    tmpz = np.where(zsum>0)[0]
    if len(tmpz)>0:
        zmin = tmpz[0]
        zmax = tmpz[-1]
        return (zmin, zmax)
    else:
        return (-1.,-1.)

def prepareLesionDistribInfoV2(niiLung, niiLesion, niiLungDIV2 = None, numZ = 3, threshLesion=0.5):
    # (1) load nii if input is a 'path'
    if isinstance(niiLung, str):# or isinstance(niiLung, unicode):
        niiLung = nib.load(niiLung)
    if isinstance(niiLesion, str):# or isinstance(niiLesion, unicode):
        niiLesion = nib.load(niiLesion)
    # (2) split lungs
    if niiLungDIV2 is not None:
        retMskLungDiv2 = niiLungDIV2
        retIsOk = True
    else:
        retMskLungDiv2, retIsOk = makeLungedMaskNii(niiLung)
    imgLungsDiv = niiImagePreTransform(retMskLungDiv2.get_data())
    imgMskLesion = niiImagePreTransform(niiLesion.get_data())
    # (3) increase number of slice for convenience
    numZ = numZ + 1
    # (4) calc percent of lesion volume in lung volume
    arrLbl = np.sort(np.unique(imgLungsDiv))
    # threshLesion=0.5
    # numZ = 4
    lst_lesion_id = sorted(lesion_id2name.keys())
    ret4Lung = {}
    ret4LesionsTypesByID = {}
    ret4LesionsTypesByNames = {}
    for ilbl in [1, 2]:
        if ilbl in arrLbl:
            lstLesionP = []
            mskLung = (imgLungsDiv == ilbl)
            zmin, zmax = getMinMaxLungZ(mskLung)
            arrz = np.linspace(zmin, zmax, numZ)
            mskLesion = imgMskLesion.copy()
            mskLesion[~mskLung] = 0

            mskLesionBin = (mskLesion>threshLesion)
            lst_les_by_id = []
            lst_les_by_names = []
            for zzi in range(numZ-1):
                z1 = int(arrz[zzi + 0])
                z2 = int(arrz[zzi + 1])
                volMsk = float(np.sum(mskLung[:, :, z1:z2]))
                volLesion = float(np.sum(mskLesionBin[:, :, z1:z2]))
                dct_les_by_id = {}
                dct_les_by_names = {}
                tmp_lesion_clip = mskLesion[:, :, z1:z2]
                for kk, vv in lesion_id2name.items():
                    if kk == 0: # skip background label
                        continue
                    tvol = float(np.sum(tmp_lesion_clip == kk))
                    tvol_rel = tvol / volMsk
                    dct_les_by_id[kk] = tvol_rel
                    dct_les_by_names[vv] = tvol_rel
                lst_les_by_id.append(dct_les_by_id)
                lst_les_by_names.append(dct_les_by_names)
                if volMsk<1:
                    volMsk = 1.
                lstLesionP.append(volLesion/volMsk)
            ret4Lung[ilbl] = lstLesionP
            ret4LesionsTypesByID[ilbl] = lst_les_by_id
            ret4LesionsTypesByNames[ilbl] = lst_les_by_names
        else:
            ret4Lung[ilbl] = None
            ret4LesionsTypesByID[ilbl] = None
            ret4LesionsTypesByNames[ilbl] = None
    return ret4Lung, ret4LesionsTypesByID, ret4LesionsTypesByNames

def prepareLesionDistribInfo(niiLung, niiLesion, numZ = 3, threshLesion=0.5):
    # (1) load nii if input is a 'path'
    if isinstance(niiLung, str):# or isinstance(niiLung, unicode):
        niiLung = nib.load(niiLung)
    if isinstance(niiLesion, str):# or isinstance(niiLesion, unicode):
        niiLesion = nib.load(niiLesion)
    # (2) split lungs
    retMskLungs, retIsOk = makeLungedMaskNii(niiLung)
    imgLungsDiv = niiImagePreTransform(retMskLungs.get_data())
    imgMskLesion = niiImagePreTransform(niiLesion.get_data())
    # (3) increase number of slice for convenience
    numZ = numZ + 1
    # (4) calc percent of lesion volume in lung volume
    arrLbl = np.sort(np.unique(imgLungsDiv))
    # threshLesion=0.5
    # numZ = 4
    ret4Lung = dict()
    for ilbl in [1, 2]:
        if ilbl in arrLbl:
            lstLesionP = []
            mskLung = (imgLungsDiv == ilbl)
            zmin, zmax = getMinMaxLungZ(mskLung)
            arrz = np.linspace(zmin, zmax, numZ)
            mskLesion = imgMskLesion.copy()
            mskLesion[~mskLung] = 0
            mskLesion = (mskLesion>threshLesion)
            for zzi in range(numZ-1):
                z1 = int(arrz[zzi + 0])
                z2 = int(arrz[zzi + 1])
                volMsk = float(np.sum(mskLung[:, :, z1:z2]))
                volLesion = float(np.sum(mskLesion[:, :, z1:z2]))
                if volMsk<1:
                    volMsk = 1.
                lstLesionP.append(volLesion/volMsk)
            ret4Lung[ilbl] = lstLesionP
        else:
            ret4Lung[ilbl] = None
    return ret4Lung

#############################################
def normalizeCTImage(pimg, outType = np.uint8):
    pimg = pimg.astype(np.float)
    vMin = -1000.
    vMax = +200.
    ret = 255. * (pimg - vMin) / (vMax - vMin)
    ret[ret < 0] = 0
    ret[ret > 255] = 255.
    return ret.astype(outType)

# generate preview
def makePreview4Lesion(dataImg, dataMsk, dataLes, sizPrv=256, nx=4, ny=3, pad=5, lesT=0.7):
    shpPrv = (sizPrv, sizPrv, sizPrv)
    dataImgR = resize3D(dataImg, shpPrv)
    dataMskR = resize3D(dataMsk, shpPrv)
    dataLesR = resize3D(dataLes, shpPrv)
    numXY = nx*ny - 1
    brd = 0.1
    arrZ = np.linspace(brd*sizPrv, (1.-brd)*sizPrv, numXY ).astype(np.int)
    cnt = 0
    tmpV = []
    for yy in range(ny):
        tmpH = []
        for xx in range(nx):
            if (yy==0) and (xx==0):
                timg = np.rot90(dataImgR[:, sizPrv // 2, :])
                tmsk = np.rot90(dataMskR[:, sizPrv // 2, :] > 0.1)
                tles0 = np.rot90(dataLesR[:, sizPrv // 2, :])
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

# generate preview

def genPreview2D(dataImg_, dataMsk_, dataLes_, pathPreview_, type_, sizPrv=256, nx=4, ny=3, pad=5, lesT=0.7):
    imgPreview_ = makePreview4LesionV2(dataImg_, dataMsk_, dataLes_, type_, sizPrv, nx, ny, pad, lesT)
    imgPreviewJson_ = {
        "description": "CT Lesion preview",
        "content-type": "image/jpeg",
        "xsize": imgPreview_.shape[1],
        "ysize": imgPreview_.shape[0],
        "url": os.path.basename(pathPreview_)
    }
    lst_legends = [mpatches.Patch(color=lesion_id2rgb[kk], label=vv) for kk, vv in lesion_id2name.items() if kk != 0]
    frame1 = plt.gca()
    frame1.axes.set_axis_off()
    fig = plt.gcf()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    DPI = fig.get_dpi()
    fig.set_size_inches(imgPreview_.shape[1] / float(DPI), imgPreview_.shape[0] / float(DPI))
    plt.imshow(imgPreview_)
    plt.legend(handles=lst_legends, loc='best', bbox_to_anchor=(1.0, 1.00), ncol=len(lst_legends))
    fig.savefig(pathPreview_, pad_inches=0)
    fig.clf()
    fig.clear()
    return imgPreviewJson_


def makePreview4LesionV2(dataImg, dataMsk, dataLes, type_=2, sizPrv=256, nx=4, ny=3, pad=5, lesT=0.7):

    if type_ == 4: # Maximum intensity projection
        print('Maximum intensity projection')
        new_img = sitk.GetImageFromArray(dataLes)
        voxel_ = [0, 0, 0]
        dim = 1
        voxel_[dim] = (dataLes.shape[dim] - 1) // 2
        projection = sitk.MaximumProjection(new_img, 0)

        sitk.WriteImage(projection, '/tmp/proj.nii.gz')
        img_nii = nib.load('/tmp/proj.nii.gz')
        imgV = img_nii.get_data()
        imgV = np.reshape(imgV, (imgV.shape[1], imgV.shape[2])).astype(np.uint8)
        return imgV

    shpPrv = (sizPrv, sizPrv, sizPrv)
    dataImgR = resize3D(dataImg, shpPrv)
    dataMskR = resize3D(dataMsk, shpPrv, order=0)
    dataLesR = resize3D(dataLes, shpPrv, order=0)
    numXY = nx*ny - 1
    brd = 0.1
    arrZ = np.linspace(brd*sizPrv, (1.-brd)*sizPrv, numXY ).astype(np.int)

    if type_ == 2:
        print('equidistant Z')
    if type_ == 3:
        print('selection of most severely affected slices')
        svrVals = np.zeros(sizPrv, np.float32)
        for zz in range(sizPrv):
            slice_ = deepcopy(dataLesR[ :, :, zz ])
            slice_[dataMskR[:, :, zz] == 0] = 0
            for kk, vv in lesion_id2name.items():
                if kk == 0:  # skip background label
                    continue
                if kk == 1:  # weight Foci by 0.2
                    svrVals[zz] += 0.2*float(np.sum(slice_[:] == kk))
                else:
                    svrVals[zz] += float(np.sum(slice_[:] == kk))
        zzIdx = np.argsort(svrVals)[::-1]
        arrSvrZ = np.zeros(len(arrZ), np.uint32)
        arrSvrZ[0] = zzIdx[0]
        tidx = 1
        i = 1
        while i < len(arrSvrZ):
            fnd = False
            for ti in range(i):
                if np.abs(zzIdx[tidx] - arrSvrZ[ti]) <= 5:
                    fnd = True
                    break
            if fnd == False:
                arrSvrZ[i] = zzIdx[tidx]
                i += 1
            tidx += 1
        arrSvrZ = np.sort(arrSvrZ)
        arrZ = deepcopy(arrSvrZ)

    cnt = 0
    tmpV = []
    for yy in range(ny):
        tmpH = []
        for xx in range(nx):
            if (yy==0) and (xx==0):
                timg = np.rot90(dataImgR[:, sizPrv // 2, :])
                tmsk = np.rot90(dataMskR[:, sizPrv // 2, :] > 0.1)
                tles0 = np.rot90(dataLesR[:, sizPrv // 2, :])
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
            # draw lung boundaries
            tmsk_bnd = _get_msk_bnd2(tmsk)
            timgR[tmsk_bnd > 0] = 0
            timgG[tmsk_bnd > 0] = 255
            timgB[tmsk_bnd > 0] = 0
            # timgR[tmsk>0] += 30
            # timgG[tmsk>0] += 30
            #
            timgRGB = np.dstack([timgR, timgG, timgB])
            timgRGB[timgRGB > 255] = 255
            timgRGB = get_overlay_msk(timgRGB, tles0)

            # tles = tles0.copy()
            # tles[tmsk<1] = 0
            # tlesT0 = 0.1
            # tlesT1 = lesT
            # tles /= tlesT1
            # tles[tles>1.] = 1.
            # tles[tles<tlesT0] = 0
            # tles = (255.*tles).astype(np.uint8)
            # timgR[tles>1] = tles[tles>1]
            timgRGB = (255. * timgRGB).astype(np.uint8)
            timgRGB = np.pad(timgRGB, pad_width=[[pad],[pad],[0]], mode='constant')
            timgRGB_ = deepcopy(timgRGB)
            if (yy != 0) or (xx != 0):
                cv2.putText(timgRGB_, 'Slice {}'.format(arrZ[cnt-1]), (timgRGB.shape[0] - 110, timgRGB.shape[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 10), 2)
                cv2.addWeighted(timgRGB_, 0.9, timgRGB, 1 - 0.9, 0, timgRGB)
            # tmpH.append(timgRGB[:, ::-1])
            tmpH.append(timgRGB)
        imgH = np.hstack(tmpH)
        tmpV.append(imgH)
    imgV = np.vstack(tmpV)
    return imgV

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_uids_from_json(json_filename_):
    if not os.path.exists(json_filename_):
        print('json file {} not exists'.format(json_filename_))
        return None,None,None,None
    with open(json_filename_) as f:
        data = json.load(f)
        try:
            study_id = data['study_id']
            patient_id = data['patient_id']
            study_uid = data['study_uid']
            series_uid = data['series_uid']

            return str(study_id), str(patient_id), str(study_uid), str(series_uid)
        except:
            print('can not read series_uid and study_uid from {}'.format(json_filename_))
            return None,None,None,None
    return None,None,None,None

def get_instance_uids_from_json(all_json_filename_, patient_id_, study_id_, study_uid_, series_uid_):
    if not os.path.exists(all_json_filename_):
        print('json file {} not exists'.format(all_json_filename_))
        return None, None
    ret_sopinstance_uids = {}
    ret_fnames = {}
    with open(all_json_filename_) as f:
        data = json.load(f)
        try:
            patient_data = data['patient']
            if patient_data['id'] == patient_id_:
                # print(data['imagingStudies'])
                imaging_data = data['imagingStudies']
                for study in imaging_data:
                    if study['studyUid'] == study_uid_:
                        # print(study['studyUid'])
                        for series in study['series']:
                            if series['uid'] == series_uid_:
                                instance_data = series['instance']
                                for instance in instance_data:
                                    # print(instance['uid'])
                                    # print(instance['number'])
                                    # print('')
                                    ret_sopinstance_uids[instance['number']] = instance['uid']
                                    ret_fnames[instance['number']] = instance['content']['url'].split('/')[-1]
            # print(ret_fnames)
            # print(ret_sopinstance_uids)
            return ret_sopinstance_uids, ret_fnames
        except:
            print('can not read sopinstance_uids {}'.format(all_json_filename_))
            return None, None
    return None, None


def vol2dcmRGB(rgb_vol_, rgb_spacing_, study_id_, patient_id_, study_uid_, series_uid_, series_desc_, sop_instance_uids_, fnames_, out_dcm_dirname_):
    if not os.path.exists(out_dcm_dirname_):
        mkdir_p(out_dcm_dirname_)

    if not isinstance(rgb_vol_, np.ndarray):
        print('vol2dcmRGB: input parameter should be ndarray')
        return
    if rgb_vol_.ndim != 4:
        print('vol2dcmRGB: dims should be 4')
        return
    if rgb_vol_.shape[3] != 3:
        print('vol2dcmRGB: volume should have 3 channels')
        return
    if not isinstance(rgb_vol_[0, 0, 0, 0], np.uint8):
        print('vol2dcmRGB: volume elements should be uint8 type')
        return

    nii_img_shape = rgb_vol_.shape

    print(rgb_spacing_)

    image_RGB = sitk.Image([nii_img_shape[2], nii_img_shape[1], nii_img_shape[0]], sitk.sitkVectorUInt8, 3)
    image_RGB = sitk.GetImageFromArray(rgb_vol_)

    # print(nii_img_shape)

    print('{}x{}x{}'.format(image_RGB.GetWidth(), image_RGB.GetHeight(), image_RGB.GetDepth()))

    image_RGB.SetSpacing(rgb_spacing_)

    # Write the 3D image as a series
    # IMPORTANT: There are many DICOM tags that need to be updated when you modify an
    #            original image. This is a delicate opration and requires knowlege of
    #            the DICOM standard. This example only modifies some. For a more complete
    #            list of tags that need to be modified see:
    #                           http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM
    #            If it is critical for your work to generate valid DICOM files,
    #            It is recommended to use David Clunie's Dicom3tools to validate the files
    #                           (http://www.dclunie.com/dicom3tools.html).

    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    direction = image_RGB.GetDirection()
    series_tag_values = [("0008|0031", modification_time),  # Series Time
                         ("0008|0021", modification_date),  # Series Date
                         ("0008|0008", "DERIVED\\LESIONMAP"),  # Image Type
                         ("0010|0020", patient_id_),  # Patiend ID
                         ("0020|0010", study_id_),  # Study ID
                         ("0020|000e", series_uid_),  # Series Instance UID
                         ("0020|000d", study_uid_),  # Study Instance UID
                         ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                                           direction[1], direction[4], direction[7])))),
                         ("0008|103e", series_desc_)]  # Series Description

    for i in range(image_RGB.GetDepth()):
        image_slice = image_RGB[:, :, i]

        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        # Setting the type to CT preserves the slice location.
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over

        # (0020, 0032) image position patient determines the 3D spacing between slices.
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str, image_RGB.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i+1))  # Instance Number


        sop_instance_uid = str(sop_instance_uids_[image_RGB.GetDepth() - i ])
        fname = fnames_[image_RGB.GetDepth() - i ]
        # print(sop_instance_uid)
        # print(fname)
        image_slice.SetMetaData("0008|0018", sop_instance_uid)  # set SOPInstanceUID
        # exit()

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join('{}{}.dcm'.format(out_dcm_dirname_, fname)))
        writer.Execute(image_slice)
        os.rename(os.path.join('{}{}.dcm'.format(out_dcm_dirname_, fname)),
                  os.path.join('{}{}'.format(out_dcm_dirname_, fname)))
    return


def niftii2dcm(nii_filename_, study_id_, patient_id_, study_uid_, series_uid_, series_desc_, sop_instance_uids_, fnames_, out_dcm_dirname_):
    if not os.path.exists(nii_filename_):
        print('Niftii file {} not exists'.format(nii_filename_))
        return
    if not os.path.exists(out_dcm_dirname_):
        mkdir_p(out_dcm_dirname_)

    nii_img = nib.load(nii_filename_)
    nii_img_vol = nii_img.get_data().astype(np.int16)
    nii_img_vol = nii_img_vol[:, ::-1, ::]

    nii_img_affine = nii_img.affine
    nii_img_shape = nii_img_vol.shape
    nii_img_spacing = [ np.abs(nii_img_affine[i][i]) for i in range(3) ]
    # print(nii_img_spacing)

    nii_img_vol = np.transpose(nii_img_vol, (2, 1, 0))

    new_img = sitk.GetImageFromArray(nii_img_vol)

    # print(nii_img_shape)

    print('{}x{}x{}'.format(new_img.GetWidth(), new_img.GetHeight(), new_img.GetDepth()))

    new_img.SetSpacing(nii_img_spacing)

    # Write the 3D image as a series
    # IMPORTANT: There are many DICOM tags that need to be updated when you modify an
    #            original image. This is a delicate opration and requires knowlege of
    #            the DICOM standard. This example only modifies some. For a more complete
    #            list of tags that need to be modified see:
    #                           http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM
    #            If it is critical for your work to generate valid DICOM files,
    #            It is recommended to use David Clunie's Dicom3tools to validate the files
    #                           (http://www.dclunie.com/dicom3tools.html).

    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()


    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    direction = new_img.GetDirection()
    series_tag_values = [("0008|0031", modification_time),  # Series Time
                         ("0008|0021", modification_date),  # Series Date
                         ("0008|0008", "DERIVED\\LESIONMAP"),  # Image Type
                         ("0010|0020", patient_id_),    # Patiend ID
                         ("0020|0010", study_id_),   # Study ID
                         ("0020|000e", series_uid_),  # Series Instance UID
                         ("0020|000d", study_uid_),  # Study Instance UID
                         ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                                           direction[1], direction[4], direction[7])))),
                         ("0008|103e", series_desc_)]  # Series Description

    for i in range(new_img.GetDepth()):
        image_slice = new_img[:, :, i]

        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time
        # Setting the type to CT preserves the slice location.
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over


        # (0020, 0032) image position patient determines the 3D spacing between slices.
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i+1))  # Instance Number

        sop_instance_uid = str(sop_instance_uids_[ new_img.GetDepth() - i  ])
        # print(sop_instance_uid)
        fname = fnames_[ new_img.GetDepth() - i ]
        # print(fname)
        image_slice.SetMetaData("0008|0018", sop_instance_uid) # set SOPInstanceUID

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join('{}{}.dcm'.format(out_dcm_dirname_, fname)))
        writer.Execute(image_slice)
        os.rename(os.path.join('{}{}.dcm'.format(out_dcm_dirname_, fname)),
              os.path.join('{}{}'.format(out_dcm_dirname_, fname)))
    return


def prepareCTpreview(series_):
    ct_nii_filename_ = series_.pathConvertedNifti(isRelative=False)
    if not os.path.exists(ct_nii_filename_):
        print('File {} not exists'.format(ct_nii_filename_))
        return
    # msk_nii_filename_ = os.path.dirname(ct_nii_filename_) + '/' + '.'.join(os.path.basename(ct_nii_filename_).split('.')[:-2]) + '-lesions3.nii.gz'
    msk_nii_filename_ = series_.pathPostprocLesions2(isRelative=False)

    # report_filename_ = os.path.dirname(ct_nii_filename_) + '/' + '.'.join(os.path.basename(ct_nii_filename_).split('.')[:-2]) + '-report2.json'
    report_filename_ = series_.pathPostprocReport(isRelative=False)

    if not os.path.exists(msk_nii_filename_):
        print('Lesion map file {} not exists'.format(msk_nii_filename_))
        return
    if not os.path.exists(report_filename_):
        print('Metadata file {} not exists'.format(report_filename_))
        return

    study_id, patient_id, study_uid, series_uid = get_uids_from_json(report_filename_)

    sop_instance_uids, fnames = get_instance_uids_from_json(all_json_filename_=os.path.dirname(ct_nii_filename_)+'/../info-all.json', patient_id_=patient_id, study_id_=study_id, study_uid_=study_uid, series_uid_=series_uid)
    # exit()

    original_out_dirname_ = os.path.realpath(os.path.dirname(ct_nii_filename_) + '/../../../@viewer/original/' + patient_id + '/' + study_uid + '/' + series_uid )+ '/'
    lesions_only_out_dirname_ = os.path.realpath(os.path.dirname(ct_nii_filename_) + '/../../../@viewer/lesions_only/' + patient_id + '/' + study_uid + '/' + series_uid )+ '/'
    lesions_map_out_dirname_ =os.path.realpath(os.path.dirname(ct_nii_filename_) + '/../../../@viewer/lesions_map/' + patient_id + '/' + study_uid + '/' + series_uid )+ '/'

    if os.path.exists(original_out_dirname_):
        shutil.rmtree(original_out_dirname_, ignore_errors=True)
    mkdir_p(original_out_dirname_)

    if os.path.exists(lesions_only_out_dirname_):
        shutil.rmtree(lesions_only_out_dirname_, ignore_errors=True)
    mkdir_p(lesions_only_out_dirname_)

    if os.path.exists(lesions_map_out_dirname_):
        shutil.rmtree(lesions_map_out_dirname_, ignore_errors=True)
    mkdir_p(lesions_map_out_dirname_)

    niftii2dcm(nii_filename_=ct_nii_filename_, study_id_=study_id, patient_id_=patient_id, series_uid_=series_uid, study_uid_=study_uid, series_desc_='Original CT',
               sop_instance_uids_=sop_instance_uids, fnames_=fnames, out_dcm_dirname_=original_out_dirname_)
    # niftii2dcm(nii_filename_=msk_nii_filename_, study_id_ = study_id, patient_id_=patient_id, series_uid_=series_uid, study_uid_=study_uid, out_dcm_dirname_=lesions_only_out_dirname_)

    # make mask overlay
    nii_img = nib.load(ct_nii_filename_)
    nii_img_vol = nii_img.get_data().astype(np.int16)
    nii_img_vol = nii_img_vol[:, ::-1, ::]

    nii_msk = nib.load(msk_nii_filename_)
    nii_msk_vol = nii_msk.get_data().astype(np.int16)
    nii_msk_vol = nii_msk_vol[:, ::-1, ::]

    nii_img_affine = nii_img.affine
    nii_img_shape = nii_img_vol.shape
    nii_img_spacing = [ np.abs(nii_img_affine[i][i]) for i in range(3) ]

    rgb_vol = np.zeros((nii_img_shape[0], nii_img_shape[1], nii_img_shape[2], 3), np.uint8)

    alpha = 0.5
    for zz in range(nii_img_shape[2]):
        img_slice_ = deepcopy(nii_img_vol[:, :, zz]).astype(np.float32)
        msk_slice_ = deepcopy(nii_msk_vol[:, :, zz])

        if img_slice_.max() > 1:
            img_slice_ = (img_slice_ - img_slice_.min()) / (img_slice_.max() - img_slice_.min())
        if img_slice_.ndim < 3:
            img_slice_ = np.tile(img_slice_[..., np.newaxis], 3)
        msk_bin = (msk_slice_ > 0)
        msk_rgb = np.tile(msk_bin[..., np.newaxis], 3)
        img_bg = (msk_rgb == False) * img_slice_
        ret = deepcopy(img_bg)
        for kk, vv in lesion_id2rgb.items():
            if kk < 1:
                continue
            tmp_msk = (msk_slice_ == kk)
            tmp_msk_rgb = np.tile(tmp_msk[..., np.newaxis], 3)
            tmp_img_overlay = alpha * np.array(vv) * tmp_msk_rgb
            tmp_img_original = (1 - alpha) * tmp_msk_rgb * img_slice_
            ret += tmp_img_overlay + tmp_img_original
        rgb_vol[:, :, zz] = (ret * 255).astype(np.uint8)

    rgb_vol = np.transpose(rgb_vol, (2, 1, 0, 3))

    vol2dcmRGB(rgb_vol_=rgb_vol, rgb_spacing_=nii_img_spacing, study_id_=study_id, patient_id_=patient_id, study_uid_=study_uid, series_uid_=series_uid,
               series_desc_='Lesion Map over original CT', sop_instance_uids_=sop_instance_uids, fnames_=fnames, out_dcm_dirname_=lesions_map_out_dirname_)

    # only rgb lesion map
    rgb_vol = np.zeros((nii_img_shape[0], nii_img_shape[1], nii_img_shape[2], 3), np.uint8)

    for zz in range(nii_img_shape[2]):
        img_slice_ = deepcopy(nii_img_vol[:, :, zz]).astype(np.float32)
        msk_slice_ = deepcopy(nii_msk_vol[:, :, zz])

        if img_slice_.max() > 1:
            img_slice_ = (img_slice_ - img_slice_.min()) / (img_slice_.max() - img_slice_.min())
        if img_slice_.ndim < 3:
            img_slice_ = np.tile(img_slice_[..., np.newaxis], 3)
        msk_bin = (msk_slice_ > 0)
        msk_rgb = np.tile(msk_bin[..., np.newaxis], 3)
        img_bg = 0.0 * img_slice_
        ret = deepcopy(img_bg)
        for kk, vv in lesion_id2rgb.items():
            if kk < 1:
                continue
            tmp_msk = (msk_slice_ == kk)
            tmp_msk_rgb = np.tile(tmp_msk[..., np.newaxis], 3)
            tmp_img_overlay = 1.0 * np.array(vv) * tmp_msk_rgb
            tmp_img_original = (1 - alpha) * tmp_msk_rgb * img_slice_
            ret += tmp_img_overlay
        rgb_vol[:, :, zz] = (ret * 255).astype(np.uint8)

    rgb_vol = np.transpose(rgb_vol, (2, 1, 0, 3))

    vol2dcmRGB(rgb_vol_=rgb_vol, rgb_spacing_=nii_img_spacing, study_id_=study_id, patient_id_=patient_id, study_uid_=study_uid, series_uid_=series_uid,
               series_desc_='Lesions Only', sop_instance_uids_=sop_instance_uids, fnames_=fnames, out_dcm_dirname_=lesions_only_out_dirname_)

    return

if __name__ == '__main__':
    pass
