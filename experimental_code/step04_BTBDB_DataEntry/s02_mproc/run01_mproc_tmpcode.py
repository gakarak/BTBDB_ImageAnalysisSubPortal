#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import numpy as np
import multiprocessing as mp

######################################
# TODO: remove in future
# def call_runnner(runner):
#     runner.run()

class Runner(object):
    def __call__(self, *args, **kwargs):
        self.run()
    def run(self):
        raise NotImplementedError
        # print ('pid=[%s]' % os.getpid()

class MyTask(Runner):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    def run(self):
        time.sleep(100)
        tmp = np.ones((1000,1000))
        for ii in range(10):
            tmp *= (tmp + self.param1) * self.param2
        print ('%s: p1=%s, p2=%s, <mean> = %s' % (os.getpid(), self.param1, self.param2, tmp.mean()))

class SimpleTaskManager(object):
    def __init__(self, nproc=4):
        self.nProc  = nproc
        self.pool   = mp.Pool(processes=self.nProc)
    def appendTask(self, task_function, task_args):
        self.pool.apply_async(task_function, [task_args] )
    def appendTaskCls(self, runner):
        self.pool.apply_async(runner)
    def waitAll(self):
        self.pool.close()
        self.pool.join()

######################################
def my_routine(params):
    p0 = params[0]
    time.sleep(p0)
    print ('%s : [%s]' % (type(params), p0))

######################################
if __name__ == '__main__':
    tm = SimpleTaskManager(nproc=6)
    rndSleep = [80]*12 #np.random.randint(2, 2, 12)

    for ii in rndSleep:
        print (':: [%s]' % ii)
        newRunner = MyTask(param1=np.random.rand(), param2=np.random.rand())
        tm.appendTaskCls(newRunner)

    tm.waitAll()
