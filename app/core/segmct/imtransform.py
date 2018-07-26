#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

from abc import abstractmethod
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import random
import matplotlib.pyplot as plt

import os
import skimage as sk
import skimage.io as skio
import skimage.transform
import scipy as sc
import scipy.ndimage
import scipy.interpolate
import numpy.matlib
import nibabel as nib

from scipy.ndimage.interpolation import map_coordinates

###############################################
def makeTransform(lstMat):
    ret = lstMat[0].copy()
    for xx in lstMat[1:]:
        ret = ret.dot(xx)
    return ret

###############################################
# Handmade scipy-based affine transformation functions for 2d/3d images: fuck you ITK/VTK!

# 2d-images
def affine_transformation_2d(image, pshift, protCnt, protAngle, pscale, pcropSize,
                             pshear=(0., 0.),
                             isDebug=False,
                             pmode='constant',
                             pval=0,
                             porder=3):
    """
    scipy-based 2d image transformation: for data augumentation
    :param image: input 2d-image with 1 or more channels
    :param pshift: shift of coordinates in row/column notation: (dr, dc)
    :param protCnt: rotation center in row/column notation: (r0, c0)
    :param protAngle: rotation angle (anti-clock-wise)
    :param pscale: scale transformation
    :param pcropSize: output size of cropped image (in row/col notation),
        if None, then output image shape is equal to input image shape
    :param pshear: shear-transform coefficients: (s_row, s_col)
    :param isDebug: if True - show the debug visualization
    :param pmode: parameter is equal 'mode' :parameter in scipy.ndimage.interpolation.affine_transform
    :param pval: parameter is equal 'val' :parameter in scipy.ndimage.interpolation.affine_transform
    :param porder: parameter is equal 'order' :parameter in scipy.ndimage.interpolation.affine_transform
    :return: transformed 2d image
    """
    # (1) precalc parameters
    angRad = (np.pi / 180.) * protAngle
    cosa = np.cos(angRad)
    sina = np.sin(angRad)
    # (2) prepare separate affine transformation matrices
    # (2.1) shift matrix: all matrices in row/column notation (this notation
    #       is default for numpy 2d-arrays. Do not confuse with XY-notation!)
    matShift = np.array([
        [1., 0., +pshift[0]],
        [0., 1., +pshift[1]],
        [0., 0.,         1.]
    ])
    # (2.2) shift matrices for rotation: backward and forward
    matShiftB = np.array([
        [1., 0., -protCnt[0]],
        [0., 1., -protCnt[1]],
        [0., 0.,          1.]
    ])
    matShiftF = np.array([
        [1., 0., +protCnt[0]],
        [0., 1., +protCnt[1]],
        [0., 0.,          1.]
    ])
    # (2.3) rotation matrix
    matRot = np.array([
        [+cosa, -sina, 0.],
        [+sina, +cosa, 0.],
        [0., 0.,       1.]
    ])
    # (2.4) scale matrix
    matScale = np.array([
        [pscale, 0., 0.],
        [0., pscale, 0.],
        [0., 0.,     1.]
    ])
    # (2.5) shear matrix
    matShear = np.array([
        [1., pshear[0], 0.],
        [pshear[1], 1., 0.],
        [0., 0., 1.],
    ])
    # (3) build total-matrix
    if pcropSize is None:
        # matTotal = matShiftF.dot(matRot.dot(matScale.dot(matShiftB)))
        matTotal = makeTransform([matShiftF, matRot, matShear, matScale, matShiftB])
        pcropSize = image.shape[:2]
    else:
        matShiftCrop = np.array([
            [1., 0., pcropSize[0] / 2.],
            [0., 1., pcropSize[1] / 2.],
            [0., 0.,                1.]
        ])
        # matTotal = matShiftCrop.dot(matRot.dot(matScale.dot(matShiftB)))
        matTotal = makeTransform([matShiftCrop, matRot, matShear, matScale, matShiftB])
    # (3.1) shift after rotation anf scale transformation
    matTotal = matShift.dot(matTotal)
    # (3.2) create inverted matrix for back-projected mapping
    matTotalInv = np.linalg.inv(matTotal)
    # (4) warp image with total affine-transform
    idxRC = np.indices(pcropSize).reshape(2, -1)
    idxRCH = np.insert(idxRC, 2, values=[1], axis=0)
    idxRCHT = matTotalInv.dot(idxRCH)[:2, :]
    if image.ndim>2:
        tret = []
        for ii in range(image.shape[-1]):
            tret.append(map_coordinates(image[:,:,ii], idxRCHT, order=porder, cval=pval, mode=pmode).reshape(pcropSize))
        tret = np.dstack(tret)
    else:
        tret = map_coordinates(image, idxRCHT, order=porder, cval=pval, mode=pmode).reshape(pcropSize)
    # (5)
    if isDebug:
        pcntPrj = matTotal.dot(list(protCnt) + [1])[:2]
        print (':: Total matrix:\n{0}'.format(matTotal))
        print ('---')
        print (':: Total matrix inverted:\n{0}'.format(matTotalInv))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.gcf().gca().add_artist(plt.Circle(protCnt[::-1], 5, edgecolor='r', fill=False))
        plt.title('Shift={0}, RotCenter={3}, Rot={1}, Scale={2}'.format(pshift, pscale, protAngle, protCnt))
        plt.subplot(1, 2, 2)
        plt.imshow(tret)
        plt.gcf().gca().add_artist(plt.Circle(pcntPrj[::-1], 5, edgecolor='r', fill=False))
        plt.title('Shape = {0}'.format(tret.shape))
        plt.show()
    return tret

# 3d-images
def affine_transformation_3d(image3d, pshiftXYZ, protCntXYZ, protAngleXYZ, pscaleXYZ, pcropSizeXYZ,
                             pshear=(0., 0., 0., 0., 0., 0.),
                             isRandomizeRot=False,
                             isDebug=False,
                             pmode='constant',
                             pval=0,
                             porder=0):
    """
    scipy-based 3d image transformation: for data augumentation
    :param image3d: input 3d-image with 1 or more channels (with shape like [sizX, sizY, sizZ]
            or [sizX, sizY, sizZ, num_channels])
    :param pshiftXYZ: shift of coordinates: (dx, dy, dz)
    :param protCntXYZ: rotation center: (x0, y0, z0)
    :param protAngleXYZ: rotation angle (anti-clock-wise), like: (angle_x, angle_y, angle_z)
    :param pscaleXYZ: cale transformation: (sx, sy, sz)
    :param pcropSizeXYZ: output size of cropped image, like: [outSizeX, outSizeY, outSizeZ].If None,
            then output image shape is equal to input image shape
    :param pshear: shear-transform 3D coefficients. Two possible formats:
            - 6-dimensional vector, like: (Syx, Szx, Sxy, Szy, Sxz, Syz)
            - 3x3 matrix, like:
                [  1, Syx, Szx]
                [Sxy,   1, Szy]
                [Sxz, Syz,   1]
    :param isRandomizeRot: if True - random shuffle order of X/Y/Z rotation
    :param isDebug: if True - show the debug visualization
    :param pmode: parameter is equal 'mode' :parameter in scipy.ndimage.interpolation.affine_transform
    :param pval: parameter is equal 'val' :parameter in scipy.ndimage.interpolation.affine_transform
    :param porder: parameter is equal 'order' :parameter in scipy.ndimage.interpolation.affine_transform
    :return: transformed 3d image
    """
    nshp=3
    # (1) precalc parameters
    angRadXYZ = (np.pi / 180.) * np.array(protAngleXYZ)
    cosaXYZ = np.cos(angRadXYZ)
    sinaXYZ = np.sin(angRadXYZ)
    # (2) prepare separate affine transformation matrices
    # (2.0) shift matrices
    matShiftXYZ = np.array([
        [1., 0., 0., +pshiftXYZ[0]],
        [0., 1., 0., +pshiftXYZ[1]],
        [0., 0., 1., +pshiftXYZ[2]],
        [0., 0., 0.,            1.]
    ])
    # (2.1) shift-matrices for rotation: backward and forward
    matShiftB = np.array([
        [1., 0., 0., -protCntXYZ[0]],
        [0., 1., 0., -protCntXYZ[1]],
        [0., 0., 1., -protCntXYZ[2]],
        [0., 0., 0.,          1.]
    ])
    matShiftF = np.array([
        [1., 0., 0., +protCntXYZ[0]],
        [0., 1., 0., +protCntXYZ[1]],
        [0., 0., 1., +protCntXYZ[2]],
        [0., 0., 0.,          1.]
    ])
    # (2.2) partial and full-rotation matrix
    lstMatRotXYZ = []
    for ii in range(len(angRadXYZ)):
        cosa = cosaXYZ[ii]
        sina = sinaXYZ[ii]
        if ii==0:
            # Rx
            tmat = np.array([
                [1.,    0.,    0., 0.],
                [0., +cosa, -sina, 0.],
                [0., +sina, +cosa, 0.],
                [0.,    0.,    0., 1.]
            ])
        elif ii==1:
            # Ry
            tmat = np.array([
                [+cosa,  0., +sina,  0.],
                [   0.,  1.,    0.,  0.],
                [-sina,  0., +cosa,  0.],
                [   0.,  0.,    0.,  1.]
            ])
        else:
            # Rz
            tmat = np.array([
                [+cosa, -sina,  0.,  0.],
                [+sina, +cosa,  0.,  0.],
                [   0.,    0.,  1.,  0.],
                [   0.,    0.,  0.,  1.]
            ])
        lstMatRotXYZ.append(tmat)
    if isRandomizeRot:
        random.shuffle(lstMatRotXYZ)
    matRotXYZ = lstMatRotXYZ[0].copy()
    for mm in lstMatRotXYZ[1:]:
        matRotXYZ = matRotXYZ.dot(mm)
    # (2.3) scale matrix
    sx,sy,sz = pscaleXYZ
    matScale = np.array([
        [sx, 0., 0., 0.],
        [0., sy, 0., 0.],
        [0., 0., sz, 0.],
        [0., 0., 0., 1.],
    ])
    # (2.4) shear matrix
    if pshear is not None:
        if len(pshear) == 6:
            matShear = np.array([
                [1., pshear[0], pshear[1], 0.],
                [pshear[2], 1., pshear[3], 0.],
                [pshear[4], pshear[5], 1., 0.],
                [0., 0., 0., 1.]
            ])
        else:
            matShear = np.eye(4, 4)
            pshear = np.array(pshear)
            if pshear.shape == (3, 3):
                matShear[:3, :3] = pshear
            elif pshear.shape == (4, 4):
                matShear = pshear
            else:
                raise Exception('Invalid shear-matrix format: [{0}]'.format(pshear))
    else:
        matShear = np.eye(4, 4)
    # (3) build total-matrix
    if pcropSizeXYZ is None:
        # matTotal = matShiftF.dot(matRotXYZ.dot(matScale.dot(matShiftB)))
        matTotal = makeTransform([matShiftF, matRotXYZ, matShear, matScale, matShiftB])
        pcropSizeXYZ = image3d.shape[:nshp]
    else:
        matShiftCropXYZ = np.array([
            [1., 0., 0., pcropSizeXYZ[0] / 2.],
            [0., 1., 0., pcropSizeXYZ[1] / 2.],
            [0., 0., 1., pcropSizeXYZ[2] / 2.],
            [0., 0., 0.,                   1.]
        ])
        # matTotal = matShiftCropXYZ.dot(matRotXYZ.dot(matScale.dot(matShiftB)))
        matTotal = makeTransform([matShiftCropXYZ, matRotXYZ, matShear, matScale, matShiftB])
    # (3.1) shift after rot-scale transformation
    matTotal = matShiftXYZ.dot(matTotal)
    # (3.2) invert matrix for back-projected mapping
    matTotalInv = np.linalg.inv(matTotal)
    # (4) warp image with total affine-transform
    idxXYZ = np.indices(pcropSizeXYZ).reshape(nshp, -1)
    idxXYZH = np.insert(idxXYZ, nshp, values=[1], axis=0)
    idxXYZT = matTotalInv.dot(idxXYZH)[:nshp, :]
    # (5) processing 3D layer-by-layer
    if image3d.ndim>nshp:
        tret = []
        for ii in range(image3d.shape[-1]):
            tret.append(map_coordinates(image3d[:, :, :, ii], idxXYZT, order=porder, cval=pval, mode=pmode).reshape(pcropSizeXYZ))
        tret = np.dstack(tret)
    else:
        tret = map_coordinates(image3d, idxXYZT, order=porder, cval=pval, mode=pmode).reshape(pcropSizeXYZ)
    # (6) Debug
    if isDebug:
        protCntXYZ = np.array(protCntXYZ)
        protCntXYZPrj = matTotal.dot(list(protCntXYZ) + [1])[:nshp]
        print (':: Total matrix:\n{0}'.format(matTotal))
        print ('---')
        print (':: Total matrix inverted:\n{0}'.format(matTotalInv))
        s0, s1, s2 = image3d.shape
        s0n, s1n, s2n = tret.shape
        #
        plt.subplot(3, 3, 1 + 0*nshp)
        plt.imshow(image3d[s0 / 2, :, :])
        plt.gcf().gca().add_artist(plt.Circle(protCntXYZ[[1, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 2 + 0*nshp)
        plt.imshow(image3d[:, s1 / 2, :])
        plt.gcf().gca().add_artist(plt.Circle(protCntXYZ[[0, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 3 + 0*nshp)
        plt.imshow(image3d[:, :, s2 / 2])
        plt.gcf().gca().add_artist(plt.Circle(protCntXYZ[[0, 1]], 5, edgecolor='r', fill=False))
        #
        plt.subplot(3, 3, 1 + 1*nshp)
        plt.imshow(tret[s0n / 2, :, :])
        plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[1, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 2 + 1*nshp)
        plt.imshow(tret[:, s1n / 2, :])
        plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[0, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 3 + 1*nshp)
        plt.imshow(tret[:, :, s2n / 2])
        plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[0, 1]], 5, edgecolor='r', fill=False))
        #
        plt.subplot(3, 3, 1 + 2 * nshp)
        plt.imshow(np.sum(tret, axis=0))
        plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[1, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 2 + 2 * nshp)
        plt.imshow(np.sum(tret, axis=1))
        plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[0, 2]], 5, edgecolor='r', fill=False))
        plt.subplot(3, 3, 3 + 2 * nshp)
        plt.imshow(np.sum(tret, axis=2))
        plt.gcf().gca().add_artist(plt.Circle(protCntXYZPrj[[0, 1]], 5, edgecolor='r', fill=False))
        plt.show()
    return tret

def generateDistortionMat2d(pshape, pgrid=(5, 5), prnd=0.2, isProportionalGrid=True):
    if isProportionalGrid:
        pgridVal = max(pgrid)
        if pshape[0] < pshape[1]:
            pgrid = (pgridVal, int(float(pshape[1]) * pgridVal / pshape[0]))
        else:
            pgrid = (int(float(pshape[0]) * pgridVal / pshape[1]), pgridVal)
    sizImg = pshape[:2]
    gxy = pgrid
    dxy = np.array(sizImg[::-1]) // np.array(gxy)
    rx  = np.linspace(0, sizImg[1], gxy[0])
    ry  = np.linspace(0, sizImg[0], gxy[1])
    XX, YY = np.meshgrid(rx, ry)
    rndXX = np.random.uniform(0, dxy[0] * prnd, XX.shape)
    rndYY = np.random.uniform(0, dxy[1] * prnd, YY.shape)
    XXrnd = XX.copy()
    YYrnd = YY.copy()
    XXrnd[1:-1, 1:-1] = XX[1:-1, 1:-1] + rndXX[1:-1, 1:-1]
    YYrnd[1:-1, 1:-1] = YY[1:-1, 1:-1] + rndYY[1:-1, 1:-1]
    fx = sc.interpolate.interp2d(XX, YY, XXrnd, kind='cubic')
    fy = sc.interpolate.interp2d(XX, YY, YYrnd, kind='cubic')
    rx_new = np.linspace(0, sizImg[1] - 1, sizImg[1])
    ry_new = np.linspace(0, sizImg[0] - 1, sizImg[0])
    XX_pert = fx(rx_new, ry_new)
    YY_pert = fy(rx_new, ry_new)
    shiftXYZ = np.matlib.repeat(np.expand_dims(np.stack((YY_pert, XX_pert, np.zeros(XX_pert.shape))), axis=-1), 3, axis=-1)
    return shiftXYZ

def generateDistortionMat3d(pshape, pgrid=(5, 5, 5), prnd=0.8, isProportionalGrid=True, isNormZScale = True):
    if isProportionalGrid:
        pgridVal = max(pgrid)
        pminVal  = np.min(pshape)
        pminIdx  = np.argmin(pshape)
        pgrid    = np.array([int(float(xx) * pgridVal / pminVal) for xx in pshape])
        pgrid[pminIdx] = pgridVal
    sizImg3D = pshape
    gxyz = pgrid
    dxyz  = np.array(sizImg3D) / np.array(gxyz)
    rxyz = [np.linspace(0, sizImg3D[ii], gxyz[ii]) for ii in range(3)]
    XYZ  = np.array(np.meshgrid(rxyz[0], rxyz[1], rxyz[2]))
    rndXYZ = np.random.uniform(0, dxyz[0] * prnd, XYZ.shape)
    if isNormZScale:
        rndXYZ[-1] *= float(pshape[-1])/pshape[0] #FIXME: potential bug
    XYZrnd = XYZ.copy()
    for ii in range(3):
        XYZrnd[ii][1:-1, 1:-1, 1:-1] += rndXYZ[ii][1:-1, 1:-1, 1:-1]
    #
    rxyz_new = [np.linspace(0, sizImg3D[ii] - 1, sizImg3D[ii]) for ii in range(3)]
    XYZ_new = np.array(np.meshgrid(rxyz_new[0], rxyz_new[1], rxyz_new[2]))
    # rxyz_new =

    # q = sc.interpolate.interpn(XYZ.reshape(3, -1), XYZrnd, XYZ_new.reshape(3, -1))
    q0 = sc.interpolate.interpn(rxyz, XYZrnd[0], XYZ_new.reshape(3, -1).T).reshape(XYZ_new.shape[1:])
    q1 = sc.interpolate.interpn(rxyz, XYZrnd[1], XYZ_new.reshape(3, -1).T).reshape(XYZ_new.shape[1:])
    q2 = sc.interpolate.interpn(rxyz, XYZrnd[2], XYZ_new.reshape(3, -1).T).reshape(XYZ_new.shape[1:])
    return np.stack((q0,q1,q2))

###############################################
def _draw_debug_3d_image(pimg3d, pmsk3d, ext_img3d = None, isShow = True, isNewFigure = False):
    tsiz    = pimg3d.shape[-1] // 2
    timg_2d = pimg3d[:, :, tsiz]
    tmsk_2d = pmsk3d[:, :, tsiz]
    tmsk_2d_n = (tmsk_2d - tmsk_2d.min()) / float(tmsk_2d.max() - tmsk_2d.min() + 0.001)
    timg_2d_n = (timg_2d - timg_2d.min()) / float(timg_2d.max() - timg_2d.min() + 0.001)
    if isNewFigure:
        plt.figure()
    nxy = 3 if ext_img3d is None else 4
    plt.subplot(1, nxy, 1)
    plt.imshow(timg_2d), plt.title('image')
    plt.subplot(1, nxy, 2)
    plt.imshow(tmsk_2d), plt.title('mask, unique = {}'.format(np.unique(pmsk3d)))
    plt.subplot(1, nxy, 3)
    plt.imshow(np.dstack([tmsk_2d_n, timg_2d_n, timg_2d_n]))
    plt.title('image + mask composite')
    if ext_img3d is not None:
        plt.subplot(1, nxy, 4)
        plt.imshow(ext_img3d[:, :, tsiz])
    if isShow:
        plt.show()

class Augumentor3DBasic:
    def __init__(self, prob = 0.5):
        self.prob = prob
    @abstractmethod
    def process(self, pimg3d, pmsk3d):
        raise NotImplementedError

class MultiAugumentor3D:
    def __init__(self, lst_aug = None):
        if lst_aug is None:
            self.lst_aug = []
        else:
            self.lst_aug = lst_aug
    def add_aug(self, paug):
        self.lst_aug.append(paug)
    def process(self, pimg3d, pmsk3d):
        ret_img3d, ret_msk3d = pimg3d, pmsk3d
        for aug_proc in self.lst_aug:
            ret_img3d, ret_msk3d = aug_proc.process(ret_img3d, ret_msk3d)
        return (ret_img3d, ret_msk3d)

class Augumentor3D_Identity(Augumentor3DBasic):
    def __init__(self, prob=0.5, isDebug = False):
        super().__init__(prob)
        self.isDebug = isDebug
    def process(self, pimg3d, pmsk3d):
        if isinstance(pimg3d, str):
            pimg3d = nib.load(pimg3d).get_data()
            pmsk3d = nib.load(pmsk3d).get_data()
        if self.isDebug:
            _draw_debug_3d_image(pimg3d, pmsk3d, isShow=True, isNewFigure=False)
        return pimg3d, pmsk3d

class Augumentor3DGeom_Affine(Augumentor3DBasic):
    def __init__(self, prob=0.5, dxyz = (10., 10., 10.), dangle=(-10., +10.), dscale=(0.9, 1.1),
                 dshear = 0., mode = 'nearest', order = 1, order_msk = 0, isDebug = False, isNoZAngleDisturb=True):
        super().__init__(prob)
        self.dxyz = dxyz
        self.dangle = dangle
        self.dscale = dscale
        self.dshear = dshear
        self.mode   = mode
        self.order  = order
        self.order_msk = order_msk
        self.isNoZAngleDisturb = isNoZAngleDisturb
        self.isDebug = isDebug
    def process(self, pimg3d, pmsk3d):
        if isinstance(pimg3d, str):
            pimg3d = nib.load(pimg3d).get_data()
            pmsk3d = nib.load(pmsk3d).get_data()
        assert (pimg3d.shape == pmsk3d.shape)
        assert (pmsk3d.ndim == 3)
        tprob = np.random.uniform(0.0, 1.0)
        if tprob > self.prob:
            return (pimg3d, pmsk3d)
        siz_crop = pimg3d.shape
        xyz_cnt = tuple((np.array(pimg3d.shape[:3])//2).tolist())
        xyz_shift = [np.random.uniform(-xx, +xx) for xx in self.dxyz]
        xyz_angle = np.random.uniform(self.dangle[0], self.dangle[1], 3)
        if self.isNoZAngleDisturb:
            xyz_angle[-1] = 0.
        xyz_angle = xyz_angle.tolist()
        xyz_scale = np.random.uniform(self.dscale[0], self.dscale[1], 3).tolist()
        xyz_shear = np.random.uniform(-self.dshear, +self.dshear, 6).tolist()
        #
        ret_img3d, ret_msk3d = [affine_transformation_3d(xx,
                                             pshiftXYZ=xyz_shift,
                                             protCntXYZ=xyz_cnt,
                                             protAngleXYZ=xyz_angle,
                                             pscaleXYZ=xyz_scale,
                                             pcropSizeXYZ=siz_crop,
                                             pshear = xyz_shear,
                                             pmode=self.mode, porder=xx_order) for xx, xx_order in
                                zip( [pimg3d, pmsk3d], [self.order, self.order_msk])]
        if self.isDebug:
            _draw_debug_3d_image(ret_img3d, ret_msk3d, isShow=True, isNewFigure=False)
        return ret_img3d, ret_msk3d

class Augumentor3DGeom_Distortion(Augumentor3DBasic):
    def __init__(self, prob=0.5, grid = (5, 5, 5), prnd=0.3, isProportionalGrid = False, mode = 'nearest',
                 order = 1, order_msk = 0, isDebug = False, isNormZScale = True):
        super().__init__(prob)
        self.grid  = grid
        self.prnd  = prnd
        self.mode  = mode
        self.order = order
        self.order_msk = order_msk
        self.isProportionalGrid = isProportionalGrid
        self.isNormZScale = isNormZScale
        self.isDebug = isDebug
    def process(self, pimg3d, pmsk3d):
        if isinstance(pimg3d, str):
            pimg3d = nib.load(pimg3d).get_data()
            pmsk3d = nib.load(pmsk3d).get_data()
        assert (pimg3d.shape == pmsk3d.shape)
        assert (pmsk3d.ndim == 3)
        distMat3D = generateDistortionMat3d(pimg3d.shape, prnd=self.prnd, pgrid=self.grid,
                                            isProportionalGrid=self.isProportionalGrid,
                                            isNormZScale=self.isNormZScale)
        ret_img3d = sc.ndimage.map_coordinates(pimg3d, distMat3D, mode=self.mode, order=self.order)
        ret_msk3d = sc.ndimage.map_coordinates(pmsk3d, distMat3D, mode=self.mode, order=self.order_msk)
        if self.isDebug:
            _draw_debug_3d_image(ret_img3d, ret_msk3d, isShow=True, isNewFigure=False)
        return ret_img3d, ret_msk3d

class Augumentor3DValues_GaussBlobs(Augumentor3DBasic):
    def __init__(self, prob=0.5, diap_num_blobs = (2, 5), diap_rad = (0.1, 0.4), diap_val = (200, 800), isDebug = False):
        super().__init__(prob)
        self.diap_num_blobs  = diap_num_blobs
        self.diap_rad  = diap_rad
        self.diap_val  = diap_val
        self.isDebug = isDebug
    def process(self, pimg3d, pmsk3d):
        if isinstance(pimg3d, str):
            pimg3d = nib.load(pimg3d).get_data()
            pmsk3d = nib.load(pmsk3d).get_data()
        assert (pimg3d.shape == pmsk3d.shape)
        assert (pmsk3d.ndim == 3)
        tsiz = pimg3d.shape[:3]
        lin_xyz = [np.linspace(0, xx - 1, xx) for xx in tsiz]
        XYZ = np.array(np.meshgrid(lin_xyz[0], lin_xyz[1], lin_xyz[2]))
        idx_xyz = np.where(pmsk3d > 0)
        tnum_blobs = np.random.randint(self.diap_num_blobs[0], self.diap_num_blobs[1])
        rnd_idx = np.random.randint(0, len(idx_xyz[0]), tnum_blobs)
        rnd_rad = np.random.uniform(self.diap_rad[0], self.diap_rad[1], tnum_blobs)
        rnd_val = np.random.uniform(self.diap_val[0], self.diap_val[1], tnum_blobs)
        ret_noise_gauss = None
        for ii, (iidx, iival) in enumerate(zip(rnd_idx, rnd_val)):
            tsigm_xyz = np.array(tsiz) * rnd_rad[ii]
            tpos_xyz = [xx[iidx] for xx in idx_xyz]
            tgauss = np.exp( -np.sum([0.5 * ((xyz - xyz0)/xyzs)**2 for xyz, xyz0, xyzs in zip(XYZ, tpos_xyz, tsigm_xyz)], axis=0) )
            tgauss *= (iival/tgauss.max())
            if ret_noise_gauss is None:
                ret_noise_gauss = tgauss
            else:
                ret_noise_gauss += tgauss
        ret_noise_gauss[pmsk3d < 0.1] = 0
        ret_img3d = pimg3d + ret_noise_gauss
        ret_msk3d = pmsk3d
        if self.isDebug:
            _draw_debug_3d_image(ret_img3d, ret_msk3d, ext_img3d=ret_noise_gauss, isShow=True, isNewFigure=False)
        return ret_img3d, ret_msk3d

class Augumentor3DValues_GaussNoise(Augumentor3DBasic):
    def __init__(self, prob=0.5, diap_scales = (0.1, 0.3), siz_edge_flt=3,
                 diap_mean = (300, 600), diap_sigm=(100, 200), isOnLungsMaskOnly=True, isDebug = False, bg_threshold = -3000.):
        super().__init__(prob)
        self.diap_scales  = diap_scales
        self.siz_edge_flt = siz_edge_flt
        self.diap_mean    = diap_mean
        self.diap_sigm    = diap_sigm
        self.isOnLungsMaskOnly = isOnLungsMaskOnly
        self.bg_threshold = bg_threshold
        self.isDebug = isDebug
    def process(self, pimg3d, pmsk3d):
        if isinstance(pimg3d, str):
            pimg3d = nib.load(pimg3d).get_data()
            pmsk3d = nib.load(pmsk3d).get_data()
        assert (pimg3d.shape == pmsk3d.shape)
        assert (pmsk3d.ndim == 3)
        tsiz = pimg3d.shape[:3]
        rnd_scale = np.random.uniform(self.diap_scales[0], self.diap_scales[1])
        tsiz_small = (np.array(tsiz) * rnd_scale).astype(np.int)
        rnd_mean = np.random.uniform(self.diap_mean[0], self.diap_mean[1])
        rnd_sigm = np.random.uniform(self.diap_sigm[0], self.diap_sigm[1])
        gauss_noise3d = np.random.normal(rnd_mean, rnd_sigm, tsiz_small.tolist())
        gauss_noise3d = sk.transform.resize(gauss_noise3d, tsiz, order=2)
        if self.isOnLungsMaskOnly:
            gauss_noise3d[pmsk3d<0.1] = 0
        if self.siz_edge_flt > 0:
            gauss_noise3d = scipy.ndimage.gaussian_filter(gauss_noise3d, self.siz_edge_flt)
        ret_img3d = pimg3d + gauss_noise3d
        ret_img3d[pimg3d < self.bg_threshold] = pimg3d[pimg3d < self.bg_threshold]
        ret_msk3d = pmsk3d
        if self.isDebug:
            _draw_debug_3d_image(ret_img3d, ret_msk3d, ext_img3d=gauss_noise3d, isShow=True, isNewFigure=False)
        return ret_img3d, ret_msk3d

class Augumentor3DValues_LinearNoise(Augumentor3DBasic):
    def __init__(self, prob=0.5, diap_mean = (-50, 50), diap_scale = (0.95, 1.1), isOnLungsMaskOnly=True, isDebug = False, bg_threshold = -3000.):
        super().__init__(prob)
        self.diap_mean    = diap_mean
        self.diap_scale   = diap_scale
        self.isOnLungsMaskOnly = isOnLungsMaskOnly
        self.bg_threshold = bg_threshold
        self.isDebug = isDebug
    def process(self, pimg3d, pmsk3d):
        if isinstance(pimg3d, str):
            pimg3d = nib.load(pimg3d).get_data()
            pmsk3d = nib.load(pmsk3d).get_data()
        assert (pimg3d.shape == pmsk3d.shape)
        assert (pmsk3d.ndim == 3)
        rnd_mean  = np.random.uniform(self.diap_mean[0], self.diap_mean[1])
        rnd_scale = np.random.uniform(self.diap_scale[0], self.diap_scale[1])
        ret_img3d = pimg3d.copy()
        ret_msk3d = pmsk3d
        if self.isOnLungsMaskOnly:
            ret_img3d[pmsk3d > 0] = rnd_mean + pimg3d[pmsk3d > 0] * rnd_scale
            ret_img3d[pimg3d < self.bg_threshold] = pimg3d[pimg3d < self.bg_threshold]
        else:
            ret_img3d[pimg3d > self.bg_threshold] = rnd_mean + rnd_scale * pimg3d[pimg3d > self.bg_threshold]
        if self.isDebug:
            _draw_debug_3d_image(ret_img3d, ret_msk3d, ext_img3d=pimg3d, isShow=True, isNewFigure=False)
        return ret_img3d, ret_msk3d

###############################################
if __name__ == '__main__':
    fnii_img = '/home/ar/data/crdf/data02_ct_lung_segm/data02_ct_lung_segm/data-luna16/lungs-img/id100225287222365663678666836860-256x256x64.nii.gz'
    fnii_msk = '/home/ar/data/crdf/data02_ct_lung_segm/data02_ct_lung_segm/data-luna16/lungs-msk/id100225287222365663678666836860-256x256x64.nii.gz'
    #

    inp_img = nib.load(fnii_img).get_data().astype(np.float32)
    inp_msk = nib.load(fnii_msk).get_data()
    inp_msk = ((inp_msk == 3) | (inp_msk == 4)).astype(np.float32)

    # aug_affine = Augumentor3DGeom_Affine(prob=1.0, isDebug=True)
    # img, msk = aug_affine.process(fnii_img, fnii_msk)

    # aug_distortion = Augumentor3DGeom_Distortion(prob=1.0, isDebug=True, order=1, order_msk=1)
    # img, msk = aug_distortion.process(inp_img, inp_msk)

    aug_gaussblobs = Augumentor3DValues_GaussBlobs(prob=1.0, isDebug=True)
    img, msk = aug_gaussblobs.process(inp_img, inp_msk)

    # aug_gaussnoise = Augumentor3DValues_GaussNoise(prob=1.0, isDebug=True, siz_edge_flt=1, isOnLungsMaskOnly=False)
    # img, msk = aug_gaussnoise.process(inp_img, inp_msk)

    # aug_linearnoise = Augumentor3DValues_LinearNoise(prob=1.0, isDebug=True, isOnLungsMaskOnly=False)
    # img, msk = aug_linearnoise.process(inp_img, inp_msk)

    # aug_affine = Augumentor3DGeom_Affine(prob=1.0, isDebug=True)
    # aug_gaussnoise = Augumentor3DValues_GaussNoise(prob=1.0, isDebug=True, siz_edge_flt=1)
    # aug_affine_and_gaussnoise = MultiAugumentor3D([aug_affine, aug_gaussnoise])
    # img, msk = aug_affine_and_gaussnoise.process(inp_img, inp_msk)

    print('-')

