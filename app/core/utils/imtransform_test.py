#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import unittest

from imtransform import affine_transformation_2d, affine_transformation_3d

####################################
class TestImtransform(unittest.TestCase):
    def test_image_transform_2d(self):
        import skimage.io as skio
        fimg = '../../../img/doge2.jpg'
        img = skio.imread(fimg, as_grey=False)
        siz = img.shape[:2]
        lstCntXY = [
            (58, 81),
            (47, 126),
            (181, 10)
        ]
        lstSiz = [None, (64, 64)]
        for ixy, parCntXY in enumerate(lstCntXY):
            for isiz, parSiz in enumerate(lstSiz):
                parCnt = parCntXY[::-1]
                imgA = affine_transformation_2d(image=img,
                                                pshift=(30., 0.),
                                                protCnt=parCnt,
                                                # pcnt=(0, 0),
                                                protAngle=16,
                                                pscale=0.9,
                                                pcropSize=parSiz,
                                                isDebug=True)
                print ('---------------------------------------------------------')
                print ('\t:: [{0}/{1}] * [{2}/{3}] : xy = {4}, size={5}'
                       .format(ixy, len(lstCntXY), isiz, len(lstSiz), parCnt, parSiz))

    def test_image_transform_3d(self):
        import numpy as np
        tvol = np.zeros((100, 100, 100))
        txyz = (50, 50, 50)
        tvol[txyz[0] - 20:txyz[0] + 20, txyz[1] - 20:txyz[1] + 20, txyz[2] - 20:txyz[2] + 20] = 1
        pshear = np.array([
            [1.,  0.1, 0.0, 0.],
            [0.0,  1., 0.0, 0.],
            [0.0, 0.0, 1.,  0.],
            [0.,   0., 0.,  1.]
        ])
        #
        img3dA = affine_transformation_3d(tvol,
                                          pshiftXYZ=(0., 0., 0.),
                                          protCntXYZ=(30, 30, 30),
                                          protAngleXYZ=(45. / 2., 0., 0.),
                                          pscaleXYZ=(1., 1., 1.),
                                          pcropSizeXYZ=(50, 50, 50),
                                          pshear=pshear,
                                          isRandomizeRot=False,
                                          isDebug=True)
    # def test_something(self):
    #     self.assertEqual(True, False)

####################################
if __name__ == '__main__':
    unittest.main()
    # testTransform = TestImtransform()
    # testTransform.test_image_transform_3d()
    # testTransform.test_image_transform_2d()


