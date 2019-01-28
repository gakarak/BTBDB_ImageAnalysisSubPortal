import numpy as np
import os
import errno
import nibabel as nib
# from matplotlib import pyplot as plt
import SimpleITK as sitk
import cv2
import requests
import json
import csv
from pprint import pprint
from io import StringIO
import io
from datetime import datetime
import calendar
from scipy.ndimage import interpolation
from copy import deepcopy

url_get_list = "https://data.tbportals.niaid.nih.gov/api/cases?since=2017-02-01&provider=RSPCPT&take=%d&skip=%d"
url_case_info = "https://data.tbportals.niaid.nih.gov/api/cases/%s"
# PATIENT_ID - CASE_ID - STUDY_ID - STUDY_UID - SERIES_UID - INSTANCE_UID
# url_dicom_file = "https://data.tbportals.niaid.nih.gov/patient/%s/case/%s/imaging/study/%s/%s/series/%s/%s.dcm"
url_dicom_file = "https://imagery.tbportal.org/%s/%s/%s/%s.dcm"


lesion_id2rgb = {
    0: [0, 0, 0],
    1: [1, 0, 0],
    2: [0, 1, 0],
    3: [0, 0, 1],
    4: [1, 1, 0],
    5: [0, 1, 1],
    6: [1, 0, 1],
    7: [0.7, 0.7, 0.7],
}


def gen_drr(volume_):
    dummy_ret = np.zeros((1, 1), np.int16)
    if volume_.ndim != 3:
        return dummy_ret
    if volume_.shape[0]*volume_.shape[1]*volume_.shape[2] == 0:
        return dummy_ret
    s_image = sitk.GetImageFromArray(arr=volume_)
    # proj = sitk.MeanProjection(image1=s_image, projectionDimension=1)
    proj = sitk.MaximumProjection(image1=s_image, projectionDimension=1)
    proj_np = sitk.GetArrayFromImage(image=proj)
    proj_vol = (np.transpose(proj_np, (0, 2, 1))[:, :, 0]).astype(np.int16)
    return proj_vol

def gen_drr_2(volume_):
    dummy_ret = np.zeros((1, 1), np.int16)
    if volume_.ndim != 3:
        return dummy_ret
    if volume_.shape[0]*volume_.shape[1]*volume_.shape[2] == 0:
        return dummy_ret

    proj_vol = np.zeros((volume_.shape[0], volume_.shape[2]), np.float32)
    volume_valid_ = (volume_[:] > -2000).astype(np.int8)
    volume_valid_proj_ = np.sum(volume_valid_, axis=1)
    volume_valid_proj_mask_ = (volume_valid_proj_ > 0)

    tmp_volume_ = np.multiply(volume_, volume_valid_)
    proj_vol = np.sum(tmp_volume_, axis=1)

    proj_vol[volume_valid_proj_mask_] = np.divide(proj_vol[volume_valid_proj_mask_], volume_valid_proj_[volume_valid_proj_mask_])
    return proj_vol


def gen_drr_3(volume_):
    dummy_ret = np.zeros((1, 1), np.int16)
    if volume_.ndim != 3:
        return dummy_ret
    if volume_.shape[0]*volume_.shape[1]*volume_.shape[2] == 0:
        return dummy_ret

    proj_vol = np.zeros((volume_.shape[0], volume_.shape[2]), np.float32)
    volume_valid_ = (volume_[:] > 0).astype(np.int8)
    volume_valid_proj_ = np.sum(volume_valid_, axis=1)
    volume_valid_proj_mask_ = (volume_valid_proj_ > 0)

    tmp_volume_ = np.multiply(volume_, volume_valid_)
    proj_vol = np.max(tmp_volume_, axis=1) # / volume_.shape[1]

    # proj_vol[volume_valid_proj_mask_] = np.divide(proj_vol[volume_valid_proj_mask_], volume_valid_proj_[volume_valid_proj_mask_])
    return proj_vol


def mkdir_p(path):
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise


def get_dicom_file_url(patient_id, case_id, study_id, study_uid, series_uid, instance_uid):
    # return url_dicom_file % (patient_id, case_id, study_id, study_uid, series_uid, instance_uid)
    return url_dicom_file % (patient_id, study_uid, series_uid, instance_uid)


def process_request(url_request):
    tmp_ret = requests.get(url_request)
    if tmp_ret.status_code == 200:
        return json.loads(tmp_ret._content)
    else:
        error_string = 'Error: %s' % tmp_ret._content
        print('*** ERROR: %s' % url_request)
        pprint(json.loads(tmp_ret._content))
        raise Exception(error_string)


def get_list_of_cases(ptake=1, pskip=0):
    url_request = url_get_list % (ptake, pskip)
    return process_request(url_request)


def get_case_info(condition_id):
    url_request = url_case_info % condition_id
    return process_request(url_request)


def download_dicom(request_url, auth_token=None):
    tmp_ret = requests.get(request_url, auth=auth_token, stream=True)
    if tmp_ret.status_code == 200:
        # buff = StringIO()
        buff = io.BytesIO()
        for chunk in tmp_ret.iter_content(2048):
            buff.write(chunk)
        return buff
    else:
        error_string = 'Error: %s' % tmp_ret._content
        print('*** ERROR: %s' % request_url)
        pprint(json.loads(tmp_ret._content))
        raise Exception(error_string)


def get_random_pair_tb(data_dir_, pair_list_):
    mkdir_p(data_dir_)

    request_info = get_list_of_cases()
    total_number = int(request_info['total'])

    for ii in range(total_number):
        temp_ret_short = get_list_of_cases(ptake=1, pskip=ii)
        json_info_short = temp_ret_short['results'][0]
        case_id = json_info_short['conditionId']
        patient_id = json_info_short['patientIdentifier']
        status = json_info_short['status']
        print('public patient_id = {}'.format(patient_id))
        if patient_id not in pair_list_: # for testing only
            continue
        if patient_id != '402':
            continue

        date_cxr_ = pair_list_[patient_id]['date_cxr']
        date_ctr_ = pair_list_[patient_id]['date_ctr']

        if status == 'final':
            json_info_case_all = get_case_info(condition_id=case_id)
            image_study_info = json_info_case_all['imagingStudies']
            if image_study_info is None:
                print('\t*** no image studies')
            else:
                case_out_dir = os.path.join(data_dir_, 'case-{}'.format(case_id))
                print('\tcondition_id = {}, #ImageStudies = {}'.format(case_id, len(image_study_info)))
                mkdir_p(case_out_dir)
                num_study = len(image_study_info)

                for study_idx, image_study in enumerate(image_study_info):
                    tmp_patient_id = json_info_case_all['patient']['id']
                    tmp_case_id = json_info_case_all['id']
                    tmp_image_study_id = image_study['id']
                    tmp_image_study_uid = image_study['studyUid']
                    print('\ttmp_case_id = {}'.format(tmp_case_id))
                    print('\ttmp_image_study_id = {}'.format(tmp_image_study_id))
                    print('\ttmp_image_study_uid = {}'.format(tmp_image_study_uid))
                    tmp_to_download_ = False

                    number_of_series = len(image_study['series'])
                    for series_idx, image_series in enumerate(image_study['series']):
                        tmp_image_series_uid = image_series['uid']
                        tmp_modality = image_series['modality']['code']
                        tmp_image_series_date = image_series['started']
                        tmp_datetime_object = datetime.strptime(tmp_image_series_date, '%Y-%m-%dT%H:%M:%S')
                        if tmp_modality == 'CR':
                            if tmp_datetime_object == date_cxr_:
                                print('\tFound CR')
                                print('\tModality = {}'.format(tmp_modality))
                                print('\ttmp_image_series_date = {}'.format(tmp_image_series_date))
                                print('\ttmp_datetime_object = {}'.format(tmp_datetime_object))
                                print('')
                                tmp_to_download_ = True
                        if tmp_modality == 'CT':
                            if tmp_datetime_object == date_ctr_:
                                print('\tFound CT')
                                print('\tModality = {}'.format(tmp_modality))
                                print('\ttmp_image_series_date = {}'.format(tmp_image_series_date))
                                print('\ttmp_datetime_object = {}'.format(tmp_datetime_object))
                                print('')
                                tmp_to_download_ = True

                    if tmp_to_download_:
                        for series_idx, image_series in enumerate(image_study['series']):
                            tmp_image_series_uid = image_series['uid']
                            tmp_modality = image_series['modality']['code']
                            number_of_instances = image_series['numberOfInstances']
                            image_series_out_dir_raw = '%s/study-%s/series-%s/raw' % (case_out_dir, tmp_image_study_id, tmp_image_series_uid)
                            mkdir_p(image_series_out_dir_raw)
                            # numI = len(imageSeries['instance'])
                            print('\t\t[%d/%d * %d/%d] #Series = %d ...' % (study_idx, num_study, series_idx, number_of_series, number_of_instances), end='')
                            for imageInstance in image_series['instance']:
                                tmp_instance_uid = imageInstance['uid']
                                try:
                                    instance_number = imageInstance['number']
                                except Exception as err:
                                    instance_number = 0
                                current_dicom_url = get_dicom_file_url(tmp_patient_id,
                                                                  tmp_case_id,
                                                                  tmp_image_study_id,
                                                                  tmp_image_study_uid,
                                                                  tmp_image_series_uid,
                                                                  tmp_instance_uid)
                                try:
                                    out_dcm_file = '%s/instance-%s-%s.dcm' % (image_series_out_dir_raw, tmp_modality, tmp_instance_uid)
                                    if os.path.isfile(out_dcm_file):
                                        continue
                                    data = download_dicom(current_dicom_url)
                                except Exception as err:
                                    break
                                with open(out_dcm_file, 'wb') as f:
                                    f.write(data.getvalue())
                                # print ('---')
                            # print ('\t\t[%d/%d * %d/%d] : dT ~ %0.3f (s)' % (iiStudy, numStudy, iiSeries, iiSeries, dT))
        exit()

    return None, None


if __name__ == '__main__':

    public_cases_list_ = {}
    with open('./CR_CT_desc.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['hasCXR'] == '1' and row['hasCTR'] == '1':
                date_cxr = datetime.strptime(row['DateCXR'], '%d/%m/%Y')
                date_ctr = datetime.strptime(row['DateCTR'], '%d/%m/%Y')
                # print(row['PublicCaseID'], row['LocalCaseID'], row['DateCXR'], row['DateCTR'], date_cxr, date_ctr)
                current_case_ = {}
                current_case_['date_cxr'] = date_cxr
                current_case_['date_ctr'] = date_ctr
                public_cases_list_[row['PublicCaseID']] = current_case_

    get_random_pair_tb(data_dir_='../../experimental_data/data_ct_xray_transfer/', pair_list_ =public_cases_list_)
    exit()

    ct_dirname = '../../experimental_data/data_ct_xray_transfer/'
    ct_filename_ = ct_dirname + 'CT.nii.gz'
    les_filename_ = ct_dirname + 'lesions5.nii.gz'

    # cxr_filename_1 = './1_init.png'
    # cxr_filename_2 = './2_init.png'

    ct_img = nib.load(filename=ct_filename_)
    ct_vol_ = ct_img.get_data()
    ct_affine = ct_img.affine
    spacings = [np.fabs(ct_affine[0, 0]), np.fabs(ct_affine[1, 1]), np.fabs(ct_affine[2, 2])]
    resize_factor = np.asarray([1.0, 1.0])
    resize_factor[1] = spacings[2] / spacings[0]
    ct_proj_ = (gen_drr_2(volume_=ct_vol_)).astype(np.int16)
    ct_proj_ = interpolation.zoom(ct_proj_, resize_factor, mode='nearest', prefilter=False)

    les_img = nib.load(filename=les_filename_)
    les_vol_ = les_img.get_data()
    les_affine = les_img.affine
    spacings = [np.fabs(les_affine[0, 0]), np.fabs(les_affine[1, 1]), np.fabs(les_affine[2, 2])]
    resize_factor = np.asarray([1.0, 1.0])
    resize_factor[1] = spacings[2] / spacings[0]
    # les_vol_[:] = 16.0*les_vol_[:]

    les_proj_ = gen_drr_3(volume_=les_vol_)
    les_proj_ = interpolation.zoom(les_proj_, resize_factor, mode='nearest', prefilter=False)
    les_proj_ = np.round(les_proj_).astype(np.int16)
    # les_proj_ = np.flip(les_proj_, axis=0)
    # les_proj_[les_proj_[:] > 1] = 2
    # les_msk_ = (les_proj_[:] <= 1)*(les_proj_[:] > 0)
    # les_proj_[les_msk_] = 1

    nib_image = nib.load(ct_dirname + 'cxr.nii.gz')
    cxr_proj_ = nib_image.get_data()[:, :, 0, 0].astype(np.float32)

    ct_proj_[:] = (ct_proj_[:] - np.min(ct_proj_))
   # ct_proj_[:] = (1024.0*(ct_proj_[:] - np.mean(ct_proj_)) / np.std(ct_proj_)).astype(np.int16)
    # cxr_proj_[:] = (1024.0*(cxr_proj_[:] - np.mean(cxr_proj_)) / np.std(cxr_proj_))

    #
    # ct_proj_[:] = (256.0 * (ct_proj_[:] - np.min(ct_proj_)) / (np.max(ct_proj_) - np.min(ct_proj_))).astype(np.uint8)
    # cxr_proj_[:] = (256.0 * (cxr_proj_[:] - np.min(cxr_proj_)) / (np.max(cxr_proj_) - np.min(cxr_proj_))).astype(np.uint8)
    #
    #
    res_factor = 1.0 * ct_proj_.shape[1] / cxr_proj_.shape[1]
    cxr_proj_2_x = int(res_factor * cxr_proj_.shape[0])
    cxr_proj_ = cv2.resize(cxr_proj_, dsize=(ct_proj_.shape[1], cxr_proj_2_x), interpolation=cv2.INTER_CUBIC).astype(np.int16)

    ct_proj_basename_ = './proj_1_ct.nii.gz'
    nib_image = nib.Nifti1Image(ct_proj_, np.eye(4))
    nib_image.to_filename(ct_proj_basename_)

    les_proj_basename_ = './proj_1_les.nii.gz'
    nib_image = nib.Nifti1Image(les_proj_, np.eye(4))
    nib_image.to_filename(les_proj_basename_)

    cxr_proj_basename_ = './proj_2_cxr.nii.gz'
    nib_image = nib.Nifti1Image(cxr_proj_, np.eye(4))
    nib.save(img=nib_image, filename=cxr_proj_basename_)

    paramMap = sitk.GetDefaultParameterMap("affine")
    # paramMap['Transform'] = ['BSplineTransform']
    # paramMap['NumberOfResolutions'] = ['8']
    # paramMap['FinalBSplineInterpolationOrder'] = ['2']
    # paramMap['MaximumNumberOfIterations'] = ['512']

    imageFilter = sitk.ElastixImageFilter()
    imageFilter.SetFixedImage(sitk.ReadImage(cxr_proj_basename_))
    imageFilter.SetMovingImage(sitk.ReadImage(ct_proj_basename_))
    imageFilter.SetParameterMap(paramMap)
    imageFilter.Execute()

    diffImage = imageFilter.GetResultImage()
    diffImageNP = np.transpose(sitk.GetArrayFromImage(diffImage), (1, 0))

    diff_basename = './cxr_ct_diff.nii.gz'
    nib_image = nib.Nifti1Image(diffImageNP, np.eye(4))
    nib_image.to_filename(diff_basename)

    # deform lesions projection
    transformMap = imageFilter.GetTransformParameterMap()
    les_deformed = sitk.Transformix(sitk.ReadImage(les_proj_basename_), transformMap)

    # les_deformed = transformix.Execute()
    les_deformed_NP = (np.transpose(sitk.GetArrayFromImage(les_deformed), (1, 0)))
    les_deformed_NP = np.round(les_deformed_NP).astype(np.int16)
    les_deformed_basename = './les_to_cxr_deform.nii.gz'
    nib_image = nib.Nifti1Image(les_deformed_NP, np.eye(4))
    nib_image.to_filename(les_deformed_basename)

    # rgb label
    les_shape_ = les_deformed_NP.shape
    les_deformed_NP_rgb = np.zeros((les_shape_[0], les_shape_[1], 3), np.uint8)
    for k in lesion_id2rgb:
        les_deformed_NP_rgb[les_deformed_NP[:] == k] = lesion_id2rgb[k]
    les_deformed_NP_rgb = 255*les_deformed_NP_rgb
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    les_deformed_NP_rgb = les_deformed_NP_rgb.copy().view(dtype=rgb_dtype).reshape(les_shape_)
    les_deformed_rgb_basename = './les_to_cxr_deform_rgb.nii.gz'
    nib_image = nib.Nifti1Image(les_deformed_NP_rgb, np.eye(4))
    nib_image.to_filename(les_deformed_rgb_basename)

    print('Gen DRR OK')
