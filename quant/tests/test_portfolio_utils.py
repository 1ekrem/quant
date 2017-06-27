'''
Created on 25 Jun 2017

@author: wayne
'''
import unittest
import numpy as np
import pandas as pd
from datetime import datetime as dt
from quant.lib import portfolio_utils as pu

d1 = dt(2017, 1, 1)
d2 = dt(2017, 1, 5)
d3 = dt(2017, 2, 10)
TB = pd.Series([0.] * 5, name='timeline', index=[dt(2016,12,30), dt(2017,1,2), dt(2017,1,3), dt(2017,1,4), dt(2017,1,5)])
T1 = pd.Series([0.] * 2, name='timeline', index=[dt(2016,12,26), dt(2017,1,2)])
TM = pd.Series([0.] * 2, name='timeline', index=[dt(2016,12,31), dt(2017,1,31)])
TMS = pd.Series([0.] * 3, name='timeline', index=[dt(2016,12,1), dt(2017,1,1), dt(2017, 2, 1)])
TDF = pd.DataFrame(np.tril(np.ones((5, 5))))
TDF[TDF <= 0] = np.nan
TONE = pd.DataFrame(np.ones((5, 2)))
TPARAMS = pd.DataFrame([[1., 3., 2.]] * 2, columns=['mean', 'median', 'std'])
TTWO = TONE.copy()
TTWO.iloc[:, 1] = np.nan
TSIG = pd.Series(np.arange(5) - 2.).to_frame()
TPRT = pd.Series([0., 0., 0., 1., 1.]).to_frame()


class TestGetTimeline(unittest.TestCase):

    def testRaiseWrongStartDate(self):
        self.assertRaises(AssertionError, pu.get_timeline, 0, d2, 'M')
    
    def testRaiseWrongEndDate(self):
        self.assertRaises(AssertionError, pu.get_timeline, d1, 0, 'M')
    
    def testRaiseWrongFrequency(self):
        self.assertRaises(AssertionError, pu.get_timeline, d1, d2, 'X')
    
    def testBFrequency(self):
        self.assertTrue(TB.equals(pu.get_timeline(d1, d2, 'B', 1)))

    def test1Frequency(self):
        self.assertTrue(T1.equals(pu.get_timeline(d1, d2, '1', 1)))

    def testMFrequency(self):
        self.assertTrue(TM.equals(pu.get_timeline(d1, d3, 'M', 1)))

    def testMSFrequency(self):
        self.assertTrue(TMS.equals(pu.get_timeline(d1, d3, 'MS', 1)))


class TestIgnoreInsufficientSeries(unittest.TestCase):

    def testRaiseWrongInput(self):
        self.assertRaises(AssertionError, pu.ignore_insufficient_series, TB, 5)

    def testCalculation(self):
        self.assertTrue(TDF.iloc[:, :3].equals(pu.ignore_insufficient_series(TDF, 3)))
        
    def testReturnsNone(self):
        self.assertIsNone(pu.ignore_insufficient_series(TDF, 6))


class TestGetDistributionScores(unittest.TestCase):
    
    def testCalculation(self):
        self.assertTrue((.5 * TONE).equals(pu.get_distribution_scores(TONE, TPARAMS)))

    def testDealWithNans(self):
        self.assertTrue((.5 * TTWO).equals(pu.get_distribution_scores(TTWO, TPARAMS)))


class TestSimpleLongOnly(unittest.TestCase):
    
    def testCalculation(self):
        self.assertTrue(TPRT.equals(pu.SimipleLongOnly(TSIG)))

    
if __name__ == "__main__":
    unittest.main()