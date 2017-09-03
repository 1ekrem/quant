'''
Created on Sep 3, 2017

@author: Wayne
'''
import timeit
import pandas as pd
from scipy import stats as ss
from quant.lib import machine_learning_utils as mu


def run_one():
    x = pd.Series(ss.norm.rvs(size=100))
    y = pd.Series(ss.norm.rvs(size=100))
    return mu.DummyStump2(x, y)


def run_two():
    x = pd.Series(ss.norm.rvs(size=100))
    y = pd.Series(ss.norm.rvs(size=100))
    return mu.DummyStump(x, y)


def run_timing():
    print('Old function time %f' % timeit.timeit(run_one, number=1000))
    print('New function time %f' % timeit.timeit(run_two, number=1000))
    