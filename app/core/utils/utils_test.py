#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import tempfile
import unittest
import app.core.utils.cmd as cmd
import shutil


class CoreUtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.dirWithDICOM = '../../../experimental_data/dataentry_test0/case-2c396a3e-1900-4fb4-bd3a-6763dc3f2ec0/study-dd10657e-f2c3-48ba-87d6-b5f3fc40c752/series-1.3.6.1.4.1.25403.163683357445804.6452.20140120113751.2/raw'
        self.tmpDIR = tempfile.mkdtemp(prefix='utils-')

    def test_pydcm2nii(self):
        self.assertTrue(os.path.isdir(self.dirWithDICOM))
        foutNii = '%s/test-out.nii.gz' % self.tmpDIR
        tret = cmd.pydcm2nii(self.dirWithDICOM, foutNii)
        self.assertTrue(tret)
        self.assertTrue(os.path.isfile(foutNii))

    def tearDown(self):
        shutil.rmtree(self.tmpDIR)

if __name__ == '__main__':
    unittest.main()
