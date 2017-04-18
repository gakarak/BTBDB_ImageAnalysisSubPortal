#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import unittest
import os
import dataentry

class TestDataEntry(unittest.TestCase):
    def setUp(self):
        self.dirDataDataEntry = '../../experimental_data/dataentry_test0'

    def test_dbwatcher(self):
        self.assertTrue(os.path.isdir(self.dirDataDataEntry))
        dbWatcher = dataentry.DBWatcher()
        dbWatcher.load(self.dirDataDataEntry, isDropEmpty=True, isDropBadSeries=True)
        dbWatcher.printStat()
        self.assertTrue(len(dbWatcher.cases) > 0)

if __name__ == '__main__':
    unittest.main()
