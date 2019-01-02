import numpy as np
import SimpleITK as sitk
import nibabel as nib
from matplotlib import pyplot as plt

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

	print(proj_vol.shape)

	# plt.imshow(proj_vol, interpolation='nearest', cmap='gray')
	# plt.show()

	proj_basename = './proj_nib.nii.gz'

	nib_image = nib.Nifti1Image(proj_vol, np.eye(4))
	nib_image.to_filename(proj_basename)


	# sitk.WriteImage(proj, proj_basename)


	return 0


if __name__ == '__main__':
	ct_dirname = '../../experimental_data/data_ct_xray_transfer/'
	ct_filename = ct_dirname + '1_init.nii.gz'

	ct_vol = nib.load(filename=ct_filename).get_data()
	gen_drr(ct_vol_=ct_vol)
	print('Gen DRR OK')
