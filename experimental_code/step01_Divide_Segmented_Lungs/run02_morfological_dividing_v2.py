#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib

import run00_common as comm

#############################################
if __name__ == '__main__':
    wdir = '../../experimental_data/resize-256x256x64'
    lstPathMsk = sorted(glob.glob('%s/*-msk.nii.gz' % wdir))
    numMsk = len(lstPathMsk)
    matplotlib.interactive(False)
    for ii,fmsk in enumerate(lstPathMsk):
        fmskLungsOut = '%s-lungs.nii.gz' % fmsk
        finfoOut = '%s-info.json' % fmsk
        if os.path.isfile(finfoOut):
            print ('\t***WARNING*** File [%s] exist, skip...' % finfoOut)
            #
        else:
            print ('[%d/%d] processing: %s' % (ii, numMsk, os.path.basename(fmsk)))
        tdat = nib.load(fmsk)
        timg = comm.niiImagePreTransform(tdat.get_data())
        retMskLungs, retIsOk = comm.makeLungedMask(timg)
        lungInfo = comm.prepareLungSizeInfo(retMskLungs, tdat.header)
        tdatLungs = nib.Nifti1Image(comm.niiImagePostTransform(retMskLungs).astype(tdat.get_data_dtype()), tdat.affine, header=tdat.header)
        nib.save(tdatLungs, fmskLungsOut)
        with open(finfoOut, 'w') as f:
            f.write(json.dumps(lungInfo, indent=4))
        # plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(timg[:,:,timg.shape[-1]/2])
        plt.title('Original mask')
        plt.subplot(1,2,2)
        plt.imshow(retMskLungs[:, :, retMskLungs.shape[-1] / 2])
        plt.title('Divided lungs')
        plt.show(block=False)
        plt.pause(0.1)
        ffigOut = '%s-preview.png' % fmsk
        plt.savefig(ffigOut)
