#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import json
import numpy as np
import nibabel as nib
import skimage.io as skio
import matplotlib.pyplot as plt
import app.core.preprocessing as preproc
from app.core.dataentry_v1 import DBWatcher
import common as comm

###############################
if __name__ == '__main__':
    dataDir = '../../experimental_data/dataentry_test0'
    dbWatcher = DBWatcher(pdir=dataDir)
    dbWatcher.load(dataDir, isDropEmpty=True, isDropBadSeries=True)
    print (dbWatcher.toString())
    for ii, ser in enumerate(dbWatcher.allSeries()):
        if ser.isConverted():
            # (0) prepare path-variables
            pathNii = ser.pathConvertedNifti(isRelative=False)
            pathSegmLungs = ser.pathPostprocLungs(isRelative=False)
            pathSegmLesions = ser.pathPostprocLesions(isRelative=False)
            pathPreview = ser.pathPostprocPreview(isRelative=False)
            pathReport = ser.pathPostprocReport(isRelative=False)
            # (1) At this point we need to segment Lung/Lesions, but at this script we skip this step to speedup
            # ...
            # (2) calc lesion score
            niiLung = nib.load(pathSegmLungs)
            niiLesion = nib.load(pathSegmLesions)
            retLesionScore = preproc.prepareLesionDistribInfo(niiLung, niiLesion)
            # (3) prepare short report about lungs
            niiLungDiv, _ = preproc.makeLungedMaskNii(niiLung)
            retLungInfo   = preproc.prepareLungSizeInfoNii(niiLungDiv)
            # (4) generate preview
            dataImg = preproc.normalizeCTImage(nib.load(pathNii).get_data())
            dataMsk = niiLung.get_data()
            dataLes = niiLesion.get_data()
            imgPreview = preproc.makePreview4Lesion(dataImg, dataMsk, dataLes)
            imgPreviewJson = {
                "description":  "CT Lesion preview",
                "content-type": "image/png",
                "xsize": imgPreview.shape[1],
                "ysize": imgPreview.shape[0],
                "url": os.path.basename(pathPreview)
            }
            skio.imsave(pathPreview, imgPreview)
            jsonReport = preproc.getJsonReport(series=ser,
                                       reportLesionScore=retLesionScore,
                                       reportLungs=retLungInfo,
                                       lstImgJson = [imgPreviewJson])
            with open(pathReport, 'w') as f:
                f.write(json.dumps(jsonReport, indent=4))
            print ('[%d] --> [%s] : %s' % (ii, os.path.basename(pathSegmLesions), retLesionScore))
