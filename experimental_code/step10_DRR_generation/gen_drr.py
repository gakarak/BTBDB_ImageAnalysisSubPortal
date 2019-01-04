import numpy as np

import nibabel as nib
from matplotlib import pyplot as plt
import SimpleITK as sitk

def gen_drr(ct_vol_):
    dummy_ret = np.zeros((1, 1), np.int16)
    if ct_vol_.ndim != 3:
        return dummy_ret
    if ct_vol_.shape[0]*ct_vol_.shape[1]*ct_vol_.shape[2] == 0:
        return dummy_ret
    # ret = sitk.StandardDeviationProjection(image1=ct_vol_, projectionDimension=1)

    ct_vol_ = np.transpose(ct_vol_, (2, 1, 0))

    s_image = sitk.GetImageFromArray(arr=ct_vol_)
    origin = [0, 0, 0]
    proj = sitk.MeanProjection(image1=s_image, projectionDimension=1)
    proj.SetOrigin(origin)
    proj.SetDirection(s_image.GetDirection())

    proj_vol = (np.transpose(sitk.GetArrayFromImage(image=proj), (2, 0, 1))[:, :, 0]).astype(np.int16)

    return proj_vol


if __name__ == '__main__':
    ct_dirname = '../../experimental_data/data_ct_xray_transfer/'
    ct_filename_1 = ct_dirname + '1_init.nii.gz'
    ct_filename_2 = ct_dirname + '2_init.nii.gz'

    ct_vol_1 = nib.load(filename=ct_filename_1).get_data()
    proj_vol_1 = gen_drr(ct_vol_=ct_vol_1)

    ct_vol_2 = nib.load(filename=ct_filename_2).get_data()
    proj_vol_2 = gen_drr(ct_vol_=ct_vol_2)

    # print(proj_vol_1.shape)
    # plt.imshow(proj_vol, interpolation='nearest', cmap='gray')
    # plt.show()

    proj_1_basename = './proj_1.nii.gz'
    nib_image = nib.Nifti1Image(proj_vol_1, np.eye(4))
    nib_image.to_filename(proj_1_basename)

    proj_2_basename = './proj_2.nii.gz'
    nib_image = nib.Nifti1Image(proj_vol_2, np.eye(4))
    nib_image.to_filename(proj_2_basename)

    resultImage = sitk.Elastix(sitk.ReadImage(proj_1_basename), sitk.ReadImage(proj_2_basename))

    print('Gen DRR OK')
