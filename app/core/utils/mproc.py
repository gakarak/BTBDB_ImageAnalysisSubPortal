#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
from datetime import datetime
import numpy as np
import multiprocessing as mp

######################################
class Runner(object):
    def __call__(self, *args, **kwargs):
        self.run()
    def getUniqueKey(self):
        raise NotImplementedError
    def run(self):
        raise NotImplementedError

######################################
class SimpleTaskManager(object):
    def __init__(self, nproc=1):
        self._nproc = nproc
        self._pool = mp.Pool(processes=self._nproc)
        self._poolState = dict()
    # def appendTask_Func(self, task_function, task_args):
    #     return self._pool.apply_async(task_function, [task_args])
    def appendTaskRunner(self, runner):
        tkey = runner.getUniqueKey()
        tstatus = self._pool.apply_async(runner)
        self._poolState[tkey] = tstatus
        return {
            tkey: tstatus
        }
    def has_key(self, pkey):
        return self._poolState.has_key(pkey)
    """
    :return (has_key: bool, ready: bool, success: bool)
    """
    def getStatusByKey(self, pkey):
        if self._poolState.has_key(pkey):
            # FIXME: potential bug: non-atomic get ready/success
            if self._poolState[pkey].ready():
                return (True, True, self._poolState[pkey].successful())
            else:
                return (True, False, False)
        else:
            return (False, False, False)
    def removeReadyStates(self):
        removedItems = dict()
        # (1) prepare ready()
        for k, v in self._poolState.items():
            if v.ready():
                removedItems[k] = v
        # (2) remove ready()
        for k in removedItems.keys():
            del self._poolState[k]
        return removedItems
    def getStat(self):
        numTot = len(self._poolState)
        numReady = 0
        numSuccess = 0
        for k, v in self._poolState.items():
            if v.ready():
                numReady += 1
                if v.successful():
                    numSuccess += 1
        return (numTot, numReady, numSuccess)
    def toString(self):
        numTot, numReady, numSuccess = self.getStat()
        return 'Task: tot/ready/success = {0}/{1}/{2}'.format(numTot, numReady, numSuccess)
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def waitAll(self):
        self._pool.close()
        self._pool.join()

######################################
if __name__ == '__main__':
    pass