'''
Created on 25 Jun 2017

@author: wayne
'''
import unittest
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


if __name__ == "__main__":
    unittest.main()