'''
Created on 8 Oct 2017

@author: wayne
'''
import os
import numpy as np
import pandas as pd
from quant.lib import optimization_utils as ou


def load_hedging_data():
    ans = {}
    filename = os.path.expanduser('~/TempWork/scripts/hedging.xlsx')
    ans['portfolio'] = pd.read_excel(filename, 'h')
    ans['beta'] = pd.read_excel(filename, 'beta')
    ans['instrument'] = pd.read_excel(filename, 'b')
    return ans


def get_beta_cov(beta):
    return pd.DataFrame(np.dot(beta.T, beta), index=beta.columns, columns=beta.columns)


def run_hedge():
    data = load_hedging_data()
    cov = get_beta_cov(data['beta'])
    o = ou.GradientDescentOptimizer(data['instrument'], data['portfolio'], cov)
    return o

