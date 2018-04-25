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
            'units': 'voxels',
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
            'units': 'voxels',
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

def getJsonReport(series, reportLesionScore, reportLungs, lstImgJson=[], reportLesionScoreById = None, reportLesionScoreByName = None):
    case_id = series.ptrCase.caseId()
    patient_id = series.ptrCase.patientId()
    study_uid = series.studyUID()
    series_uid = series.uid()
    retLesions = {}
    # retLesions = {
    #     'left': None,
    #     'right': None
    # }
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
    ret = {
        'case_id' : case_id,
        'patient_id' : patient_id,
        'study_uid' : study_uid,
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
def makePreview4LesionV2(dataImg, dataMsk, dataLes, sizPrv=256, nx=4, ny=3, pad=5, lesT=0.7):
    shpPrv = (sizPrv, sizPrv, sizPrv)
    dataImgR = resize3D(dataImg, shpPrv)
    dataMskR = resize3D(dataMsk, shpPrv, order=0)
    dataLesR = resize3D(dataLes, shpPrv, order=0)
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
            tmpH.append(timgRGB)
        imgH = np.hstack(tmpH)
        tmpV.append(imgH)
    imgV = np.vstack(tmpV)
    return imgV

if __name__ == '__main__':
    pass