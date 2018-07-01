'''
Created on 27 Sep 2017

@author: wayne
'''
import pandas as pd
import numpy as np
from quant.lib import portfolio_utils as pu
from quant.data import stocks
from datetime import datetime as dt


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


def get_momentum(rtns, vol, mom):
    return get_clean_returns(rtns).rolling(mom, min_periods=1).sum() / vol / np.sqrt(mom)


def get_signal_1(rtns, vol):
    '''
    Momentum
    '''
    s1 = get_momentum(rtns, vol, 3)
    s2 = get_momentum(rtns, vol, 52).shift(3)
    sig = s2 - s1
    return sig.apply(get_top_3, axis=1)


def get_signal_2(rtns, vol):
    '''
    Reversal
    '''
    s1 = get_momentum(rtns, vol, 2)
    s2 = get_momentum(rtns, vol, 26).shift(2)
    sig = .3 * s2 - s1
    return sig.apply(get_top_3, axis=1)


def get_pnl(sig, rtns, vol):
    pos = (1. / vol)[sig > 0].ffill(limit=4)
    return rtns.mul(pos.shift()).sum(axis=1)
    

def plot(rtns, vol, posvol):
    sig = get_signal_1(rtns, vol)
    get_pnl(sig, rtns, posvol).cumsum().plot()


def get_smx_data():
    u = stocks.get_ftse_smx_universe()
    r = stocks.load_google_stock_returns(data_name='Returns')
    r = r.resample('W').sum()
    v = r.abs().rolling(52, min_periods=13).median()
    v2 = v.copy()
    v2[v2 < .03] = .03
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
    sig = pd.concat([get_momentum(r, v, 2).iloc[-1], get_momentum(r, v, 52).shift(2).iloc[-1], capital / posvol.iloc[-1]], axis=1)
    sig.columns = ['Reversal', 'M52', 'Position']
    sig = sig.sort_values('M52', ascending=False)
    sig = sig.dropna()
    return sig, pos.iloc[-1] * capital, pos2.iloc[-1] * capital, sig_date, pnl, pnl2

