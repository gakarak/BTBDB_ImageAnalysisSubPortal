# coding: utf-8

import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial.distance import *
import cv2
from copy import deepcopy
import json

root_data_path = '../data/bel_crdf/'
les_relative_path = 'lesions_relabelled/'
sgm_relative_path = 'regsegm_nii/'

les_suffix = '_lesions'
seg_suffix = '_regsegm'

do_cbir = False

segments = [2, 3] # разделение лёгкого на левое и правое, а также каждая половина - на 3 сегмента по оси z
num_of_segments = 1
for s in segments:
    num_of_segments = num_of_segments * s

lesions_classes = { '1': 'Foci',
                    '2': 'Caverns',
                    '3': 'Fibrosis',
                    '4': 'Plevritis',
                    '5': 'Atelectasis',
                    '6': 'Pneumothorax'}
                    # '6': 'Pneumathorax'}
# lesions_classes = { '1': 'class_1', '2': 'class_2', '3': 'class_3', '4': 'class_4', '5': 'class_5', '6':'class_6'}
lesions_classes2 = {vv:kk for kk,vv in lesions_classes.items()}
lesions_weights = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0]

# 1     1       1   0   0
# 2     1       1   0   0
# 3     1       1   0   0
# 4     1       1   0   0
# 5     1       1   0   0
# 6     2       0   0   0
# 7     3       0   0.6 0.2
# 8     4       0.2 0.6 0.2
# 9     5       0.8 0.6 0.2
# 10    6       0.6 0.6 0

def get_segment(sgm_filename_, pos_vector_):
    sgm_img = nib.load(sgm_filename_).get_data()

    return sgm_img


def get_left_right_lungs(sgm_img_):
    shape_ = sgm_img_.shape
    min_val = np.min(sgm_img_)
    max_val = np.max(sgm_img_)
    middle_index = 0
    min_nonzero_count = shape_[0] * shape_[1]
    for c in range(shape_[0] // 2, shape_[0] // 2 + 50):
        seg_slice = sgm_img_[c, :, :]
        nonzero_count = np.sum(seg_slice > min_val)
        if nonzero_count < min_nonzero_count:
            min_nonzero_count = nonzero_count
            middle_index = c

    for c in range(shape_[0] // 2 - 50, shape_[0] // 2):
        seg_slice = sgm_img_[c, :, :]
        nonzero_count = np.sum(seg_slice > min_val)
        if nonzero_count < min_nonzero_count:
            min_nonzero_count = nonzero_count
            middle_index = c

    left_img = deepcopy(sgm_img_)
    left_img[middle_index:, :, :] = min_val

    right_img = deepcopy(sgm_img_)
    right_img[:middle_index, :, :] = min_val

    return left_img, right_img


def get_left_right_lungs_ar(sgm_img_):
    shape_ = sgm_img_.shape
    min_val = np.min(sgm_img_)

    left_img = deepcopy(sgm_img_)
    left_img[left_img[:] !=2 ] = min_val

    right_img = deepcopy(sgm_img_)
    right_img[right_img[:] !=1] = min_val

    return left_img, right_img


def get_lung_part(sgm_img_, total_parts_, part_index_):
    dp_ = 100.0/total_parts_
    if part_index_ < 0:
        part_index_ = 0
    if part_index_ > total_parts_ - 1:
        part_index_ = total_parts_ - 1
    sgm_img_t = deepcopy(sgm_img_)

    min_val = np.min(sgm_img_t)
    sgm_img_t -= min_val

    sgm_img_t[sgm_img_t[:] > 0] = 1
    nonzero_ind = np.nonzero(sgm_img_t)

    low_limit = int(part_index_*dp_)
    up_limit = int((part_index_+1)*dp_)

    low_ind_ = int(np.percentile(nonzero_ind[2], low_limit)) + 1
    up_ind_ = int(np.percentile(nonzero_ind[2], up_limit)) + 1

    sgm_img_t[:, :, :low_ind_] = 0
    sgm_img_t[:, :, up_ind_:] = 0

    return np.nonzero(sgm_img_t)


# sgm_filename_ - путь к файлу с сегментированным легким
# les_filename_ - путь к файлу с маской lesions
# shad_idx - идентификатор случая, строка
def calc_desc(sgm_filename_, les_filename_, shad_idx_='dummy_idx'):
    if os.path.isfile(les_filename_) and os.path.isfile(sgm_filename_):
        print(les_filename_)
        sgm_img = nib.load(sgm_filename_).get_data()
        les_img = nib.load(les_filename_).get_data()
        # left_lung, right_lung = get_left_right_lungs(sgm_img)
        left_lung, right_lung = get_left_right_lungs_ar(sgm_img)

        # for y in range(sgm_img.shape[1]):
        #     im_to_save = sgm_img[:, y, :]
        #     im_to_save = cv2.normalize(im_to_save, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        #     cv2.imwrite('./tmp/sgm_'+str(y)+'.png', im_to_save)

        min_v = np.min(les_img)
        max_v = np.max(les_img)
        les_img -= min_v

        for val in range(1, 6):
            les_img[les_img[:] == val] = 1
        for val in range(6, 11):
            les_img[les_img[:] == val] = val - 4
        #
        # for y in range(les_img.shape[1]):
        #     im_to_save = les_img[:, y, :]
        #     im_to_save = cv2.normalize(im_to_save, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        #     cv2.imwrite('./tmp/les_'+str(y)+'.png', im_to_save)

        # exit()

        left_parts = []
        right_parts = []

        for i in range(3):
            part_l = get_lung_part(sgm_img_=left_lung, total_parts_=3, part_index_=i)
            left_parts.append(part_l)
            part_r = get_lung_part(sgm_img_=right_lung, total_parts_=3, part_index_=i)
            right_parts.append(part_r)

        desc_ = np.zeros((segments[0], segments[1], len(lesions_classes)), np.float32)

        # left lung
        for p in range(3):
            les_img_f = deepcopy(les_img[left_parts[p]])
            sum_les_img_f = np.sum((les_img_f[:] == 1).astype(np.uint8))
            sum_left_parts = len(left_parts[p][0])

            foci_percent = sum_les_img_f / sum_left_parts
            # print('left: ' + str(p) + ': ' + str(foci_percent))
            desc_[0, p, 0] = foci_percent

            for l in range(2, 7):
                les_val = np.sum((les_img_f[:] == l).astype(np.uint8))
                if les_val > 64:
                    desc_[0, p, l - 1] = 1

        # right lung
        for p in range(3):
            les_img_f = deepcopy(les_img[right_parts[p]])
            sum_les_img_f = np.sum((les_img_f[:] == 1).astype(np.uint8))
            sum_right_parts = len(right_parts[p][0])

            foci_percent = sum_les_img_f / sum_right_parts
            # print('right: ' + str(p) + ': ' + str(foci_percent))
            desc_[1, p, 0] = foci_percent

            for l in range(2, 7):
                les_val = np.sum((les_img_f[:] == l).astype(np.uint8))
                if les_val > 64:
                    desc_[1, p, l - 1] = 1

        return desc_

def recalc_descriptors(lesions_pairs_set_filenames_, out_desc_filename_):
    # count_ = 0
    # if os.path.exists(out_desc_filename_):
    #     os.remove(out_desc_filename_)
    # with open(lesions_pairs_set_filenames_) as pairs_file:
    #     pairs_reader = csv.DictReader(pairs_file, delimiter='|')
    #     for row in pairs_reader:
    #         les_filename = row['les_filename']
    #         sgm_filename = row['sgm_filename']
    #         shad_idx = row['ID']
    #         desc = calc_desc(sgm_filename_=sgm_filename, les_filename_=les_filename, shad_idx_=shad_idx)
    #         with open(out_desc_filename_, 'a') as f:
    #             f.write(str(shad_idx))
    #             for lr in range(desc_.shape[0]):
    #                 for p in range(desc_.shape[1]):
    #                     for les in range(desc_.shape[2]):
    #                         f.write('|' + '{:10.6f}'.format(desc_[lr, p, les]))
    #             f.write('\n')
    #         print('\n')
    pass

def load_descriptors(desc_filename_):
    desc_list = []
    shad_idx_list = []

    with open(desc_filename_) as desc_file:
        desc_reader = csv.reader(desc_file, delimiter='|')
        for row in desc_reader:
            desc_ = np.zeros((segments[0], segments[1], len(lesions_classes)), np.float32)
            shad_idx_list.append(row[0])
            idx = 1
            for lr in range(desc_.shape[0]):
                for p in range(desc_.shape[1]):
                    for les in range(desc_.shape[2]):
                        desc_[lr, p, les] = row[idx]
                        idx += 1
            desc_list.append(desc_)
    return shad_idx_list, desc_list


def desc_to_json(desc_):
    lung_side_by_name = ["left_by_name", "right_by_name"]
    lung_side_by_id = ["left_by_id", "right_by_id"]
    lung_side_int = ["left", "right"]

    desc_json = {}
    desc_json["lesions"] = {}
    for lr in range(desc_.shape[0]):
            desc_json["lesions"][lung_side_by_name[lr]] = []
    for lr in range(desc_.shape[0]):
        for p in range(desc_.shape[1]):
            desc_json["lesions"][lung_side_by_name[lr]].append({})
    for lr in range(desc_.shape[0]):
        for p in range(desc_.shape[1]):
            for les in range(desc_.shape[2]):
                desc_json["lesions"][lung_side_by_name[lr]][p][lesions_classes[str(les+1)]] = float(desc_[lr][p][les])

    for lr in range(desc_.shape[0]):
        desc_json["lesions"][lung_side_by_id[lr]] = []
    for lr in range(desc_.shape[0]):
        for p in range(desc_.shape[1]):
            desc_json["lesions"][lung_side_by_id[lr]].append({})
    for lr in range(desc_.shape[0]):
        for p in range(desc_.shape[1]):
            for les in range(desc_.shape[2]):
                desc_json["lesions"][lung_side_by_id[lr]][p][str(les + 1)] = float(
                    desc_[lr][p][les])

    for lr in range(desc_.shape[0]):
        desc_json["lesions"][lung_side_int[lr]] = []
        for p in range(desc_.shape[1]):
            desc_json["lesions"][lung_side_int[lr]].append(float(
                    desc_[lr][p][0]))

    # return json.dumps(desc_json, indent=4, sort_keys=True)
    return desc_json


def desc_from_json_file(json_filename_):
    cur_desc = np.zeros((segments[0], segments[1], len(lesions_classes)), np.float32)
    if os.path.isfile(json_filename_):
        with open(json_filename_) as case_json:
            case_json_data = json.load(case_json)
            cur_desc = desc_from_json_object(case_json_data)
    return cur_desc


def desc_in_json_file(json_filename_):
    if os.path.isfile(json_filename_) == False:
        return False
    with open(json_filename_) as case_json:
        case_json_data = json.load(case_json)
        return desc_in_json_object(case_json_data)


def desc_in_json_object(json_object_):
    if 'lesions' in json_object_:
        return True
    return False


def desc_from_json_object(json_object_):
    desc_ = np.zeros((segments[0], segments[1], len(lesions_classes)), np.float32)
    if desc_in_json_object(json_object_):
        for key, value in json_object_['lesions'].items():
            if key == 'left_by_name':
                p = 0
                if value is not None:
                    for v in value:
                        for vv in v:
                            # les_ind = int(lesions_classes.keys()[lesions_classes.values().index(vv)]) - 1
                            les_ind = int(lesions_classes2[vv]) - 1
                            desc_[0, p, les_ind] = v[vv]
                        p += 1
            if key == 'right_by_name':
                p = 0
                if value is not None:
                    for v in value:
                        for vv in v:
                            les_ind = int(lesions_classes2[vv]) - 1
                            # les_ind = int(lesions_classes.keys()[lesions_classes.values().index(vv)]) - 1
                            desc_[1, p, les_ind] = v[vv]
                        p += 1
    return desc_


def desc_dist_q(a, b, metrics_): # quick distance, only by availability of lesions in lungs

    a_ = deepcopy(a)
    b_ = deepcopy(b)
    for lr in range(a.shape[0]):
        for p in range(a.shape[1]):
            if a[lr, p, 0] > 0.01:
                a_[lr, p, 0] = 1
            else:
                a_[lr, p, 0] = 0
            if b[lr, p, 0] > 0.01:
                b_[lr, p, 0] = 1
            else:
                b_[lr, p, 0] = 0
    aa = np.sum(a_, axis=(0, 1))
    bb = np.sum(b_, axis=(0, 1))
    aa[aa > 0] = 1
    bb[bb > 0] = 1

    return desc_distance(a = aa, b = bb, metrics_=metrics_)


def desc_dist_p(a, b, metrics_): # distance, which takes into account the position of lesion in lungs
    a_ = deepcopy(a)
    b_ = deepcopy(b)

    a_parts = a_.reshape((a_.shape[0] * a_.shape[1], a_.shape[2]))
    b_parts = b_.reshape((b_.shape[0] * b_.shape[1], b_.shape[2]))

    #FIXME: check NaN-removing code:
    a_parts[np.isnan(a_parts)] = -1
    b_parts[np.isnan(b_parts)] = -1
    a_parts[np.isinf(a_parts)] = -1
    b_parts[np.isinf(b_parts)] = -1

    diff_by_lesions = np.zeros((a_parts.shape[1]), np.float32)

    # try:
    for les in range(diff_by_lesions.shape[0]):
        diff_by_lesions[ les ] = desc_distance(a_parts[:, les], b_parts[:, les], metrics_=metrics_)
    # except Exception as err:
    #     print('[{}]'.format(err))
    #     print('-')

    integral_diff = np.sum(np.multiply(diff_by_lesions, lesions_weights))

    return integral_diff, diff_by_lesions


def desc_distance(a, b, metrics_):
    if metrics_ == 'euclidean':
        return euclidean(a, b)
    if metrics_ == 'correlation':
        return correlation(a, b)
    if metrics_ == 'linalg':
        return np.linalg.norm(a - b)
    if metrics_ == 'sqrt':
        return np.sqrt(np.linalg.norm(np.multiply(a - b, a - b)))
    return 0


def calc_diff_matrices(desc_list_, metrics_='euclidean', ext_data = None):

    desc_list_len = len(desc_list_)

    diff_matrix_by_presence = np.zeros((desc_list_len, desc_list_len), np.float32)
    diff_matrix_integral = np.zeros((desc_list_len, desc_list_len), np.float32)
    diff_matrix_by_lesion = np.zeros((desc_list_len, desc_list_len, len(lesions_classes)), np.float32)

    for i in range(desc_list_len):
        for j in range(i, desc_list_len):
            if ext_data is not None:
                print(' processing: [{} * {}] -> {}/{}'.format(i, j, ext_data[i], ext_data[j]))
            diff_int, diff_by_les = desc_dist_p(desc_list_[i], desc_list_[j], metrics_)
            diff_matrix_integral[ i, j ] = diff_int
            diff_matrix_by_lesion[ i, j ] = diff_by_les

            diff_pres = desc_dist_q(desc_list_[i], desc_list_[j], metrics_)
            diff_matrix_by_presence[ i, j ] = diff_pres

    for i in range(desc_list_len):
        for j in range(i):
            diff_matrix_integral[i, j] = diff_matrix_integral[j, i]
            diff_matrix_by_lesion[i, j] = diff_matrix_by_lesion[j, i]
            diff_matrix_by_presence[i, j] = diff_matrix_by_presence[j, i]

    diff_matrix_by_lesion = np.transpose(diff_matrix_by_lesion, axes=(2, 0, 1))

    return diff_matrix_by_presence, diff_matrix_integral, diff_matrix_by_lesion


def make_cbir(desc_ind_, diff_matrix_, knn_number_=3, thresh_=0.0):
    dist_list = np.zeros((diff_matrix_.shape[0] - 1, 2), np.float32)

    dist_list[:, 0] = [i for i in range(diff_matrix_.shape[0]) if i != desc_ind_]
    dist_list[:, 1] = [x for i, x in enumerate(diff_matrix_[:, desc_ind_]) if i != desc_ind_]

    dist_sorted = sorted(dist_list, key=lambda x: x[1])

    nearest_desc_indices = [int(x[0]) for x in dist_sorted[:knn_number_]]
    nearest_desc_distances = [x[1] for x in dist_sorted[:knn_number_]]

    if thresh_ > 0:
        nearest_desc_distances = [x for x in nearest_desc_distances if x < thresh_]
        nearest_desc_indices = nearest_desc_indices[0:len(nearest_desc_distances)]

    return nearest_desc_indices, nearest_desc_distances


def recalc_diff_matrices(desc_list_, pres_mat_filename_='', int_mat_filename_='', les_mat_filename_=''):
    pres_m, int_m, les_m = calc_diff_matrices(desc_list_)

    if pres_mat_filename_ != '':
        np.savetxt(pres_mat_filename_, pres_m, fmt='%10.5f')
    if int_mat_filename_ != '':
        np.savetxt(int_mat_filename_, int_m, fmt='%10.5f')
    if les_mat_filename_ != '':
        for les in range(len(lesions_classes)):
            np.savetxt(les_mat_filename_ + str(les) + '.txt', les_m[les], fmt='%10.5f')
    return 0


if __name__ == '__main__':

    do_recalc_descriptors = False
    do_recalc_matrices = False

    lesions_weights = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0]

    if do_recalc_descriptors:
        do_recalc_descriptors(lesions_pairs_set_filenames_='./lesions_pairs_set.csv', out_desc_filename_='./desc.csv')

    shad_idx, desc_list = load_descriptors(desc_filename_='./desc.csv')

    if do_recalc_matrices:
        recalc_diff_matrices(desc_list_=desc_list)

    pres_m = 0
    int_m = 0
    les_m = 0

    case_json_name = 'series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2-CT-report.json'

    if desc_in_json_file(case_json_name)== False:
        print('Section "lesions" is not available in '+case_json_name)
        exit()
    desc_loaded_from_json = desc_from_json_file(case_json_name)
    print(desc_loaded_from_json)
    # check if available in list
    desc_found = False
    d_ind = -1
    for d in desc_list:
        diff_int, diff_vec = desc_dist_p(desc_loaded_from_json, d, 'euclidean')
        d_ind += 1
        if diff_int < 0.000001:
            print('found')
            desc_found = True
            break

    if desc_found == False:
        desc_list_tmp = deepcopy(desc_list)
        desc_list_tmp.append(desc_loaded_from_json)
        recalc_diff_matrices(desc_list_tmp, './pres_mat_tmp.txt', './int_mat_tmp.txt', './les_mat_tmp_')
        d_ind = len(desc_list_tmp) - 1
        pres_m = np.loadtxt('./pres_mat_tmp.txt')
        int_m = np.loadtxt('./int_mat_tmp.txt')
        les_m = np.zeros((len(lesions_classes), int_m.shape[0], int_m.shape[1]), np.float32)
        for les in range(len(lesions_classes)):
            les_m[les] = np.loadtxt('./les_mat_tmp_' + str(les) + '.txt')
    else:
        pres_m = np.loadtxt('./pres_mat.txt')
        int_m = np.loadtxt('./int_mat.txt')
        les_m = np.zeros((len(lesions_classes), int_m.shape[0], int_m.shape[1]), np.float32)
        for les in range(len(lesions_classes)):
            les_m[les] = np.loadtxt('./les_mat_' + str(les) + '.txt')
    # exit()

    n_ind, n_dist = make_cbir(desc_ind_=d_ind, diff_matrix_=int_m, knn_number_=3)
    print('#: '+str(n_ind))
    n_shad_idx = [shad_idx[i] for i in n_ind]
    print('ids: '+str(n_shad_idx))
    print('distances: '+str(n_dist))
    print('\n')
    print (desc_list[n_ind[0]])
