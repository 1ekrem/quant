'''
Created on 8 Oct 2017

@author: wayne
'''
import unittest
import numpy as np
import pandas as pd
from quant.lib import optimization_utils as ou

a = [1., np.nan]
ax = np.array([[1.], [0.]])
b = [1., np.inf]
c = np.array([1., 0.])
d = pd.DataFrame([[1., 0.]])

da = pd.Series([1., np.nan])
dax = pd.DataFrame([[1.], [0.]])
db = pd.Series([1., np.inf])
dc = pd.Series([1., 0.])

w = np.array([[1.], [1.]])
p = np.array([[3.], [3.]])
v = np.array([[1., 0.5], [0.5, 1.]])
ins = np.array([[1., 1.9, -0.2], [0.8, -0.5, 1.1]])
stk = np.array([[0.5], [0.5], [0.5]])
s = np.array([[2.3], [1.9], [1.4]])

portfolio = ou.Portfolio(w, ins, stk)
variance = ou.Variance(w, v)
    

class TestGetWeightMatrix(unittest.TestCase):

    def testFillNan(self):
        return self.assertTrue(np.allclose(ax, ou.get_weight_matrix(a)))

    def testFillInf(self):
        return self.assertTrue(np.allclose(ax, ou.get_weight_matrix(b)))

    def testArray(self):
        return self.assertTrue(np.allclose(ax, ou.get_weight_matrix(c)))

    def testPandas(self):
        return self.assertTrue(np.allclose(ax, ou.get_weight_matrix(d)))


class TestGetWeightDataFrame(unittest.TestCase):

    def testFillNan(self):
        return self.assertTrue(dax.equals(ou.get_weight_dataframe(da)))
    
    def testFillInf(self):
        return self.assertTrue(dax.equals(ou.get_weight_dataframe(db)))
    
    def testPandas(self):
        return self.assertTrue(dax.equals(ou.get_weight_dataframe(dc)))
    

class TestVariance(unittest.TestCase):
        
    def testCalc(self):
        return self.assertEqual(3., variance.result())
    
    def testPrime(self):
        return self.assertTrue(np.allclose(p, variance.prime()))


class TestPortfolio(unittest.TestCase):
    
    def testCalc(self):
        return self.assertTrue(np.allclose(s, portfolio.result()))
    
    def testPrime(self):
        return self.assertTrue(np.allclose(ins.T, portfolio.prime()))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()