'''
Created on 23 Jun 2017

@author: wayne
'''
import unittest
import numpy as np
import pandas as pd
from quant.lib import machine_learning_utils as mu


a = np.arange(10.)
b = np.array([-1.] * 3 + [1.] * 7)
b2 = np.array([-1.] * 4 + [1.] * 7)
c = np.array([-1.] * 6 + [1.] * 4)
d = pd.Series(a)
dnan = d.copy()
dnan.iloc[0] = np.nan
e = pd.Series(b).to_frame()
enan = e.copy()
enan.iloc[0, 0] = np.nan
efillna = e.copy()
efillna.iloc[0, 0] = 0.
e2 = pd.Series(b2).to_frame()
f = pd.concat([pd.Series(b), pd.Series(c)], axis=1)


class TestGiveMePandasVariables(unittest.TestCase):

    def testRaiseAssertError1(self):
        self.assertRaises(AssertionError, mu.give_me_pandas_variables, a, e)
    
    def testRaiseAssertError2(self):
        self.assertRaises(AssertionError, mu.give_me_pandas_variables, d, b)
    
    def testRaiseAssertError3(self):
        self.assertRaises(AssertionError, mu.give_me_pandas_variables, d, e2)
    
    def testOutputTransformation(self):
        self.assertFalse(d.equals(mu.give_me_pandas_variables(d, e)[0]))
    
    def testOutput1(self):
        self.assertTrue(d.to_frame().equals(mu.give_me_pandas_variables(d, e)[0]))
    
    def testOutput2(self):
        self.assertTrue(e.equals(mu.give_me_pandas_variables(d, e)[1]))

    def testOutput1fillna(self):
        self.assertTrue(d.to_frame().equals(mu.give_me_pandas_variables(dnan, e)[0]))

    def testOutput2fillna(self):
        self.assertTrue(efillna.equals(mu.give_me_pandas_variables(d, enan)[1]))


class TestPositiveSum(unittest.TestCase):
    
    def testPositiveOutcome(self):
        self.assertEqual(7., mu.positive_sum(pd.Series(b)))

    def testZeroOutcome(self):
        self.assertEqual(0., mu.positive_sum(pd.Series(np.zeros(5))))


class TestNegativeSum(unittest.TestCase):
    
    def testNegativeOutcome(self):
        self.assertEqual(-3., mu.negative_sum(pd.Series(b)))

    def testZeroOutcome(self):
        self.assertEqual(0., mu.negative_sum(pd.Series(np.zeros(5))))


class TestStumpError(unittest.TestCase):
    
    def testZeroPrediction(self):
        self.assertTrue(np.isnan(mu.StumpError(d, pd.Series(np.zeros(10)), 1.)))
    
    def testNanInput(self):
        self.assertTrue(np.isnan(mu.StumpError(d, pd.Series(b), np.nan)))
    
    def testPerfectPrediction(self):
        self.assertEqual(0., mu.StumpError(d, pd.Series(b), 3.))

    def testImperfectPrediction(self):
        self.assertEqual(0.1, mu.StumpError(d, pd.Series(b), 4.))


class TestDummyStump(unittest.TestCase):
    
    def testEqualOutcome(self):
        self.assertEqual(10., mu.DummyStump(d, pd.Series(np.zeros(10)))[0])

    def testEqualInput(self):
        self.assertEqual(0., mu.DummyStump(0. * d, pd.Series(b))[0])

    def testCalculation(self):
        self.assertEqual(2.5, mu.DummyStump(d, pd.Series(b))[0])


class TestStumpPrediction(unittest.TestCase):
    
    def testCalculation(self):
        self.assertTrue(e.equals(mu.StumpPrediction(d.to_frame(), 3.)))


if __name__ == "__main__":
    unittest.main()
