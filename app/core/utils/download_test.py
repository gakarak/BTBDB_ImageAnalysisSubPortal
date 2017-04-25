#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import unittest
import download as dwd

import tempfile
import shutil

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix='BTBDB')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_something(self):
        runnerDataEntry = dwd.RunnerDataEntry(data_dir=self.temp_dir)
        runnerDataEntry.refreshCases()
        self.assertTrue(runnerDataEntry.numCases()>0)

if __name__ == '__main__':
    unittest.main()
