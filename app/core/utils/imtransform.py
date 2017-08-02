#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

from scipy.interpolate import RegularGridInterpolator
import numpy as np
import random
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import map_coordinates

#######################################
def makeTransform(lstMat):
    ret = lstMat[0].copy()
    for xx in lstMat[1:]:
        ret = ret.dot(xx)
    return ret

#######################################
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

#############################
if __name__ == '__main__':
    pass