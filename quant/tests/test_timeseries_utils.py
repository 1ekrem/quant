'''
Created on Jun 1, 2017

@author: Wayne
'''
import unittest
import pandas as pd
import numpy as np
from datetime import datetime as dt
from quant.lib import timeseries_utils as tu

a = pd.Series([1., 2.], index=[dt(2010,1,1), dt(2010,2,1)])
b = pd.DataFrame([[0., 1.], [2., 3.]], index=[dt(2010,1,1), dt(2010,1,15)], columns=['C1', 'C2'])
c = pd.Series([1., 1.], index=[dt(2010,1,1), dt(2010,1,15)])
d = pd.Series([1., np.nan], index=[dt(2010,1,1), dt(2010,1,15)])
x = np.arange(3)
TONE = pd.DataFrame(np.ones((5, 2)))
TPARAMS = pd.DataFrame([[1., 3., 2.]] * 2, columns=['mean', 'median', 'std'])
TTWO = TONE.copy()
TTWO.iloc[:, 1] = np.nan


class TestResample(unittest.TestCase):

    def testRaiseAssertError1(self):
        self.assertRaises(AssertionError, tu.resample, x, a)

    def testRaiseAssertError2(self):
        self.assertRaises(AssertionError, tu.resample, a, x)
    
    def testResampleWithForefilling(self):
        self.assertTrue(c.equals(tu.resample(a, b, True)))

    def testResampleWithoutForefilling(self):
        self.assertTrue(d.equals(tu.resample(a, b, False)))


class TestGetDistributionScores(unittest.TestCase):
    
    def testCalculation(self):
        self.assertTrue((.5 * TONE).equals(tu.get_distribution_scores(TONE, TPARAMS)))

    def testDealWithNans(self):
        self.assertTrue((.5 * TTWO).equals(tu.get_distribution_scores(TTWO, TPARAMS)))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()