import os
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# DATA_DIR_NAME = 'data-real'
# DATA_DIR_NAME = 'experimental_data/dataentry_test0'
# DATA_DIR_NAME = '/home/ar/data2/crdf/test_dataentry0'
# DATA_DIR_NAME = '/home/ar/data2/crdf/@Data_BTBDB_ImageAnalysisSubPortal_test'
# DATA_DIR_NAME = '/mnt/s3-imlab.tbportal.org/test_dataentry0'
DATA_DIR_NAME = os.path.join(str(Path.home()), 'data', 'crdf', '@Data_BTBDB_ImageAnalysisSubPortal_s3')

class Config(object):
    DEBUG = True
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = 'this-really-needs-to-be-changed'
    #
    DIR_BASE = BASE_DIR
    DIR_DATA = os.path.join(BASE_DIR, DATA_DIR_NAME)
    # case-id, stupy-id, series-id, modality-typy(str), file-final part
    URL_S3_CLOUD_FRONT = 'https://imlab.tbportal.org'
    URL_TEMPLATE_S3_REL = 'crdf/@Data_BTBDB_ImageAnalysisSubPortal_s3/case-{}/study-{}/series-{}-{}-{}'
    if not os.path.isdir(DIR_DATA):
        os.makedirs(DIR_DATA)
    def get_s3_cloud_front_url(self, rel_url):
        return os.path.join(self.URL_S3_CLOUD_FRONT, rel_url)

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True