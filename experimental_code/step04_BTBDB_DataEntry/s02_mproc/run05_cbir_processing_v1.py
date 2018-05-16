#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import time
import app
import json
import app.backend
from app.core.dataentry_v1 import DBWatcher
from app.core.utils import log as log
from app.core.utils.report import RunnerMakeReport
import app.core.lesion_descriptors as ldsc
import logging
import concurrent.futures as mp

def _task_evaluate_ldsc(pdata):
    path_sgm = pdata[0]
    path_les = pdata[1]
    path_jsn = pdata[2]
    #
    if ldsc.desc_in_json_file(path_jsn):
        logging.debug(' [*] found CBIR descriptor in file, loading from [{}]'.format(path_jsn))
        cur_desc = ldsc.desc_from_json_file(path_jsn)
    else:
        logging.debug(' [*] CBIR descriptor not found, precalculation from: [{}/{}]'.format(path_sgm, path_les))
        cur_desc = ldsc.calc_desc(sgm_filename, les_filename)
        json_data = None
        with open(path_jsn, 'r') as f:
            dsc_json = ldsc.desc_to_json(cur_desc)
            json_data = json.loads(f.read())
            json_data['lesions'] = dsc_json
        if json_data is not None:
            with open(path_jsn, 'w') as f:
                f.write(json.dumps(json_data, indent=4))
    tret = (path_jsn, cur_desc)
    return tret

def _task_simple_cbir(pdata):
    dsc_idx   = pdata[0]
    path_json = pdata[1]
    dsc       = pdata[2]
    int_m     = pdata[3]
    lst_fjson = pdata[4]
    logging.debug(' [CBIR] --> [{}] * [{}] * dsc-shape = {}'.format(dsc_idx, path_json, dsc.shape))
    n_ind, n_dist = ldsc.make_cbir(dsc_idx, diff_matrix_=int_m, knn_number_=2)
    #
    n_shad_idx = [lst_fjson[i] for i in n_ind]
    case_json = open(lst_fjson[dsc_idx], 'r')
    case_json_data = json.load(case_json)
    case_json.close()
    case_json_data['similar_cases'] = []
    for n_idx in n_shad_idx:
        with open(n_idx) as n_case_json:
            n_case_json_data = json.load(n_case_json)
            cur_similar_json_data = {}
            cur_similar_json_data['case_id']        = n_case_json_data['case_id']
            cur_similar_json_data['series_uid']     = n_case_json_data['series_uid']
            cur_similar_json_data['study_uid']      = n_case_json_data['study_uid']
            cur_similar_json_data['study_id']       = n_case_json_data['study_id']
            cur_similar_json_data['preview_images'] = n_case_json_data['preview_images']
            case_json_data['similar_cases'].append(cur_similar_json_data)
    case_json = open(json_filename_list[dsc_idx], 'w')
    case_json.write(json.dumps(case_json_data, indent=4, sort_keys=True))
    case_json.close()
    return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # data_dir = 'data-real'
    num_threads = 6
    data_dir = app.backend.config.DIR_DATA
    db_watcher = DBWatcher(pdir=data_dir)
    ptr_log = log.get_logger(data_dir)
    print('{}'.format(db_watcher))
    lst_task_data_ldsc = []
    for ii, ser in enumerate(db_watcher.allSeries()):
        if ser.isPostprocessed():
            sgm_filename = ser.pathPostprocLungsDiv2(isRelative=False)
            les_filename = ser.pathPostprocLesions2(isRelative=False)
            json_filename = ser.pathPostprocReport(isRelative=False)
            if (sgm_filename is not None) and (les_filename is not None) and (json_filename is not None):
                tdata = [sgm_filename,les_filename, json_filename]
                lst_task_data_ldsc.append(tdata)
                logging.info('\tadd series [{}] for calculation CBIR-descriptor'.format(ser))
            else:
                logging.warning(' [!!!] cant find valid input data for series [{}]: {}'.format(ser, json_filename))
    #
    logging.info(' [*] pre-calculating lesion-cbir descriptors: #tasks/#threads = {}/{}'.format(len(lst_task_data_ldsc), num_threads))
    # lst_ret_ldsc = []
    # for xx in lst_task_data_ldsc:
    #     tret = _task_evaluate_ldsc(xx)
    #     lst_ret_ldsc.append(tret)
    t1 = time.time()
    tpool = mp.ProcessPoolExecutor(max_workers=num_threads)
    lst_ret_ldsc = tpool.map(_task_evaluate_ldsc, lst_task_data_ldsc)
    lst_ret_ldsc = [xx for xx in lst_ret_ldsc if xx is not None]
    tpool.shutdown(wait=True)
    dt = time.time() - t1
    logging.info('\tDone (dsc)! dt ~ {:0.1f} (s)'.format(dt))
    #
    desc_list = [xx[-1] for xx in lst_ret_ldsc]
    json_filename_list = [xx[0] for xx in lst_ret_ldsc]
    t1 = time.time()
    logging.info(' [*] build distance matrix')
    # pres_m, int_m, les_m = ldsc.calc_diff_matrices(desc_list_=desc_list, metrics_='euclidean')
    _, int_m, _ = ldsc.calc_diff_matrices(desc_list_=desc_list, metrics_='euclidean')
    dt = time.time() - t1
    logging.info('\tDone (dist-mat)! dt ~ {:0.1f} (s)'.format(dt))
    #
    lst_task_data_cbir = [(xxi, xx[0], xx[1], int_m, json_filename_list) for xxi, xx in enumerate(lst_ret_ldsc)]
    logging.info(' [*] find similar cases for every series: #tasks/#threads = {}/{}'.format(len(lst_task_data_cbir), num_threads))
    t1 = time.time()
    lst_ret_cbir = []
    for xx in lst_task_data_cbir:
        tret = _task_simple_cbir(xx)
        lst_ret_cbir.append(tret)
    dt = time.time() - t1
    logging.info('\tDone (cbir)! dt ~ {:0.1f} (s)'.format(dt))
    print('-')
    # pathModelLung = '../../../experimental_data/models/fcnn_ct_lung_segm_2.5d_tf/'
    # pathModelLesion = '../../../experimental_data/models/fcnn_ct_lesion_segm_3d_tf/'
    # runnerMakeReport = RunnerMakeReport(data_dir=data_dir,
    #                                     dirModelLung=pathModelLung,
    #                                     dirModelLesion=pathModelLesion,
    #                                     listGpuId=[1])
    # #FIXME: this runner can be runned acynchronously over mproc.SimpleTaskManager
    # runnerMakeReport.run()
    print ('---')