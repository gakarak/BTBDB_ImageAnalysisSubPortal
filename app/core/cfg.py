#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os

URL_S3_CLOUD_FRONT = 'https://imlab.tbportal.org'
URL_TEMPLATE_S3_REL = 'crdf/@Data_BTBDB_ImageAnalysisSubPortal_s3/case-{}/study-{}/{}'


def get_s3_cloud_front_url(rel_url):
    return os.path.join(URL_S3_CLOUD_FRONT, rel_url)

if __name__ == '__main__':
    pass