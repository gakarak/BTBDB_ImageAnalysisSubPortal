#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import matplotlib.pyplot as plt
import nibabel as nib

import run00_common as comm

#############################################
if __name__ == '__main__':
    fimg = '../../experimental_data/resize-256x256x64/0009-256x256x64-msk.nii.gz'
    foutLungs = '%s-lungs.nii.gz' % fimg
    tdat = nib.load(fimg)
    timg = comm.niiImagePreTransform(tdat.get_data())
    retMskLungs, retIsOk = comm.makeLungedMask(timg, isDebug=True)
    lungInfo = comm.prepareLungSizeInfo(retMskLungs, tdat.header)
    tdatLungs = nib.Nifti1Image(comm.niiImagePreTransform(retMskLungs).astype(tdat.get_data_dtype()), tdat.affine, header=tdat.header)
    nib.save(tdatLungs, foutLungs)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(timg[:,:,timg.shape[-1]/2])
    plt.subplot(1,2,2)
    plt.imshow(retMskLungs[:, :, retMskLungs.shape[-1] / 2])
    plt.show()
