'''
Created on 27 Sep 2017

@author: wayne
'''
import pandas as pd
import numpy as np
from quant.lib import portfolio_utils as pu
from quant.data import stocks
from quant.research import cross
from datetime import datetime as dt

STOCK_VOL_FLOOR = 0.02


def get_a_signal(r, rm, i=3):
    '''
    3nd Gen reversal signal
    '''
    s1 = cross.get_stock_mom(rm, i)
    s2 = cross.get_stock_mom(rm, 52).shift(i)
    #acc = r.cumsum().ffill()
    #ax = acc.rolling(i, min_periods=1).min()
    sig = 1. * (s1 < -.5) * (s2 > .5)# * (acc == ax)
    return sig[sig > 0]


def get_smx_data():
    u = stocks.get_ftse_smx_universe()
    r = stocks.load_google_returns(data_name='Returns', data_table=stocks.UK_STOCKS)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rm, vol = cross.get_returns(r)
    return rtn, rm, vol


def get_ftse250_data():
    u = stocks.get_ftse250_universe()
    r = stocks.load_google_returns(data_name='Returns', data_table=stocks.UK_STOCKS)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rm, vol = cross.get_returns(r)
    return rtn, rm, vol


def get_a_bundle(r, rm, posvol, capital, i):
    pos = get_a_signal(r, rm, i)
    v = pos.sum(axis=1).mean()
    sig_date = pos.index[-1]
    acc = r.cumsum()
    dd = acc.rolling(13, min_periods=1).max() - acc
    pos = (1. / posvol)[pos > 0].ffill(limit=3)[dd >= .09]
    pnl = r.mul(pos.shift())
    pnl_idx = r.mul(1. / posvol.shift())
    p1 = pd.concat([pos.iloc[-1], pnl.iloc[-1]], axis=1) * capital
    p1.columns = ['Position', 'PnL']
    p1 = p1.loc[p1.Position.abs() > 0]
    pnl = pnl.sum(axis=1)
    pnl.name = 'A%d' % i
    pnl_idx = pnl_idx.mean(axis=1) * v
    pnl_idx.name = 'Index'
    return sig_date, p1, pnl, pnl_idx
    
    
def run_package(r, rm, posvol, capital=500):
    pos = []
    pnls = []
    for i in [3, 6, 9]:
        sig_date, p1, pnl, pnl_idx = get_a_bundle(r, rm, posvol, capital, i)
        pos.append(p1)
        pnls.append(pnl)
    pnls.append(pnl_idx)
    return sig_date, pos, pnls
