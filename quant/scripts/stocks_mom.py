'''
Created on 27 Sep 2017

@author: wayne
'''
import pandas as pd
import numpy as np
from quant.lib import portfolio_utils as pu
from quant.data import stocks
from datetime import datetime as dt

STOCK_VOL_FLOOR = 0.02


def get_top_3(x):
    ans = np.nan * x
    s = np.argsort(x[-x.isnull()])
    if len(s) > 3:
        s = s[-3:]
    ans.loc[x[-x.isnull()].iloc[s].index] = 1.
    return ans


def get_clean_returns(rtns):
    r = rtns.copy()
    r[r.abs() > .7] = np.nan
    return r


def get_signal_1(rtns, vol):
    '''
    Reversal
    '''
    r = rtns.divide(vol)
    rm = r.subtract(r.mean(axis=1), axis=0)
    s1 = rm.rolling(3).mean()
    s2 = rm.rolling(52, min_periods=13).mean().shift(3)
    sig = 1. * ((s1 <= -.6) & (s2 >= .4))
    return sig[sig.abs() > 0]


def get_signal_2(rtns, vol):
    '''
    Reversal
    '''
    r = rtns.divide(vol)
    rm = r.subtract(r.mean(axis=1), axis=0)
    s1 = rm
    s2 = rm.rolling(52, min_periods=13).mean().shift()
    sig = 1. * ((s1 <= -2.) & (s2 >= .3))
    return sig[sig.abs() > 0]


def get_pnl(sig, rtns, vol):
    pos = (1. / vol)[sig > 0].ffill(limit=4)
    return rtns.mul(pos.shift()).sum(axis=1)
    

def plot(rtns, vol, posvol):
    sig = get_signal_2(rtns, vol)
    get_pnl(sig, rtns, posvol).cumsum().plot()


def get_smx_data():
    u = stocks.get_ftse_smx_universe()
    r = stocks.load_google_returns(data_name='Returns', data_table=stocks.UK_STOCKS)
    r = r.loc[:, r.columns.isin(u.index)]
    r = r.resample('W').sum()
    w = r.abs()
    v = w[w > 0].rolling(52, min_periods=13).median().ffill().bfill()
    v2 = v.copy()
    v2[v2 < STOCK_VOL_FLOOR] = STOCK_VOL_FLOOR
    return r.loc[:, u.index], v.loc[:, u.index], v2.loc[:, u.index]


def run_new_smx(r, v, posvol, capital=500):
    mom = get_signal_1(r, v)
    rev = get_signal_2(r, v)
    sig_date = mom.index[-1]
    pos = mom.apply(get_top_3, axis=1)
    pos2 = rev.apply(get_top_3, axis=1)
    pos = (1. / posvol)[pos > 0].ffill(limit=4)
    pos2 = (1. / posvol)[pos2 > 0].ffill(limit=4)
    pnl = get_clean_returns(r).mul(pos.shift()).sum(axis=1)
    pnl2 = get_clean_returns(r).mul(pos2.shift()).sum(axis=1)
    rtn = r.divide(v)
    rtn = rtn.subtract(rtn.mean(axis=1), axis=0)
    s1 = rtn.rolling(3).mean()
    s2 = rtn.rolling(52, min_periods=13).mean().shift(3)
    s3 = rtn.rolling(2).mean()
    s4 = rtn.rolling(52, min_periods=13).mean().shift(2)
    s5 = rtn.rolling(6).mean()
    s6 = rtn.rolling(52, min_periods=13).mean().shift(6)
    sig = pd.concat([s1.iloc[-1], s2.iloc[-1], s3.iloc[-1], s4.iloc[-1], s5.iloc[-1], 
                     s6.iloc[-1], capital / posvol.iloc[-1]], axis=1)
    sig.columns = ['Low3', 'High3', 'Low2', 'High2', 'Low6', 'High6', 'Position']
    sig = sig.sort_values('High3', ascending=False)
    sig = sig.dropna()
    return sig, pos.iloc[-1] * capital, pos2.iloc[-1] * capital, sig_date, pnl, pnl2

