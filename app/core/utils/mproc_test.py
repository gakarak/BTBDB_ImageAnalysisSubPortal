import unittest

import os
import time
from datetime import datetime
import numpy as np
from mproc import SimpleProcessManager, ProcessRunner

######################################
class MyTask(ProcessRunner):
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
        print ('%s: p1=%s, p2=%s, dt=%s' % (os.getpid(), self.param1, self.param2, self.dt))

######################################
class TestMProc(unittest.TestCase):
    def test_simple_taskrunner(self):
        tm = SimpleProcessManager(nproc=3)
        numTasks = 12
        rndSleep = np.random.randint(1, 3, numTasks)
        for ii in rndSleep:
            newRunner = MyTask(param1=np.random.rand(), param2=np.random.rand(), dt=ii)
            print (':: [{0}] : {1}'.format(ii, newRunner.getUniqueKey()))
            tm.appendTaskRunner(newRunner)
        tm.waitAll()
        numTot, numReady, numSuccess = tm.getStat()
        self.assertEqual(numTot,     numTasks)
        self.assertEqual(numReady,   numTasks)
        self.assertEqual(numSuccess, numTasks)

######################################
if __name__ == '__main__':
    unittest.main()
