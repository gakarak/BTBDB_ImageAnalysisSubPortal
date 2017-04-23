#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import enum
import json

"""
status for stage/total: 
    - new : all stages is incompleted
    - ok  : is finished
    - in  : in progress
    - err : error, and
"""
class StageStatus(enum.Enum):
    STATE_NEW   = 'new'
    STATE_OK    = 'ok'
    STATE_IN    = 'in'
    STATE_ERR   = 'err'
    STATES = [STATE_NEW, STATE_OK, STATE_IN, STATE_ERR]
    def __init__(self, pdict=None):
        if pdict is None:
            self._state = StageStatus.STATE_NEW
            self._serr = None
        else:
            if 'state' in pdict:
                pass
            else:
                self._state = StageStatus.STATE_ERR
                self._serr = ''

class SeriesProcStages:
    FILE_STATUS = 'status.json'
    def __init__(self, pdir=None):
        pass
    def buildNew(self):
        ret = dict()

if __name__ == '__main__':
    pass