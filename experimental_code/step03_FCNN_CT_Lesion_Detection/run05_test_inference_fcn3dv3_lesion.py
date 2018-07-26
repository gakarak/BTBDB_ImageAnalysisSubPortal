#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import app.core.segmct.fcnn_lesion3dv3 as fcn3dv3

# class Args:
#     mode_run    = 'infer'
#     mode_cls    = 'mclass'
#     crop3d      = '96x96x64'
#     is_zerobg   = False
#     infer_shape = '512x512x256'
#     infer_crop  = '128x128x64'
#     infer_pad   = 32
#     idx_trn     = None
#     idx_val     = None
#     model       = None
#     img         = None
#     lung        = None
#     infer_out   = None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path_img = '../../experimental_data/test_cases_for_segm/TB1_004_11053_1_16777217.nii.gz'
    path_msk_lungs = '../../experimental_data/test_cases_for_segm/TB1_004_11053_1_16777217.nii.gz-lungs.nii.gz'
    path_msk_out = '{}-lesionv3.nii.gz'
    path_model = '../../experimental_data/models/fcn3d_ct_lesion_segm_v3_tf/idx-all.txt-trn-s1.txt_fcn3d_valloss_96x96x64_bgFalse.h5'
    print('-')
    args = fcn3dv3.get_args_obj()
    args.model  = path_model
    args.img    = path_img
    args.lung   = path_msk_lungs
    args.infer_out = path_msk_out
    #
    cfg = fcn3dv3.Config(args)
    t1 = time.time()
    model = fcn3dv3.build_model(cfg, inp_shape=list(cfg.infer_crop_pad) + [1])
    model.summary()
    logging.info('\tloading model weights from [{}]'.format(cfg.path_model))
    model.load_weights(cfg.path_model, by_name=True)
    ret_cls, ret_pmap = fcn3dv3.run_inference(cfg, model, path_out_nii=cfg.infer_out)
    dt = time.time() - t1
    logging.info('\t\tdone... dt ~ {:0.3f} (s) * [{}]'.format(dt, cfg.path_img))