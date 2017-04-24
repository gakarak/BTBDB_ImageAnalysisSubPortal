#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
from datetime import datetime
import numpy as np
import multiprocessing as mp

######################################
# TODO: remove in future
# def call_runnner(runner):
#     runner.run()

class Runner(object):
    def __call__(self, *args, **kwargs):
        self.run()
    def getUniqueKey(self):
        raise NotImplementedError
    def run(self):
        raise NotImplementedError
        # print ('pid=[%s]' % os.getpid()

class MyTask(Runner):
    def __init__(self, param1, param2, dt=None):
        self.param1 = param1
        self.param2 = param2
        if dt is not None:
            self.dt = dt
        else:
            self.dt = 100
    def getUniqueKey(self):
        return 'runner-{0}-{1}'.format(os.getpid(), datetime.now().strftime('%Y.%m.%d-%H.%M.%S:%f'))
    def run(self):
        time.sleep(self.dt)
        tmp = np.ones((1000,1000))
        for ii in range(10):
            tmp *= (tmp + self.param1) * self.param2
        print ('%s: p1=%s, p2=%s, <mean> = %s' % (os.getpid(), self.param1, self.param2, tmp.mean()))

class SimpleTaskManager(object):
    def __init__(self, nproc=4):
        self._nproc  = nproc
        self._pool   = mp.Pool(processes=self._nproc)
        self._poolState = dict()
    def appendTask_BK(self, task_function, task_args):
        return self._pool.apply_async(task_function, [task_args])
    def appendTaskRunner(self, runner):
        tkey    = runner.getUniqueKey()
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
    def toString(self):
        numTot = len(self._poolState)
        numReady = 0
        numSuccess = 0
        for k,v in self._poolState.items():
            if v.ready():
                numReady += 1
                if v.successful():
                    numSuccess += 1
        return 'Task: tot/ready/success = {0}/{1}/{2}'.format(numTot, numReady, numSuccess)
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def waitAll(self):
        self._pool.close()
        self._pool.join()

######################################
def my_routine(params):
    p0 = params[0]
    time.sleep(p0)
    print ('%s : [%s]' % (type(params), p0))

######################################
if __name__ == '__main__':
    tm = SimpleTaskManager(nproc=6)
    rndSleep = [20]*12 #np.random.randint(2, 2, 12)
    for ii in rndSleep:
        print (':: [%s]' % ii)
        newRunner = MyTask(param1=np.random.rand(), param2=np.random.rand(), dt=ii)
        tm.appendTaskRunner(newRunner)

    tm.waitAll()
