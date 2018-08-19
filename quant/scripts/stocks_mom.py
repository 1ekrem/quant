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


def get_b_signal(rtns, vol, s, i=3, low=1., high=.5):
    '''
    Type B reversal signal
    '''
    r = rtns.divide(vol)
    rm = r.subtract(r.mean(axis=1), axis=0)
    s1 = rm.rolling(i, min_periods=1).sum() / np.sqrt(1. * i)
    s2 = rm.rolling(52, min_periods=13).sum().shift(i) / np.sqrt(52.)
    sig = 1. * ((s1 <= -low) & (s2 >= high))
    return sig[(sig.abs() > 0) & (s <= .2)]


def get_rev_signal_v2(rtns, vol, s, i=3, low=1., high=.5):
    '''
    2nd Gen reversal signal
    '''
    r = rtns.divide(vol)
    rm = r.subtract(r.mean(axis=1), axis=0)
    s1 = rm.rolling(i, min_periods=1).mean()
    s2 = rm.rolling(52, min_periods=13).mean().shift(i)
    acc = r.cumsum().ffill()
    ax = acc.rolling(i, min_periods=1).min()
    sig = 1. * ((s1 <= -low) & (s2 >= high) & (acc == ax))
    return sig[(sig.abs() > 0) & (s <= .2)]


def get_signal_1(rtns, vol, s):
    '''
    Reversal 6 weeks
    '''
    return get_rev_signal_v2(rtns, vol, s, 6, .5, .3)


def get_signal_2(rtns, vol, s):
    '''
    reversal 7 weeks
    '''
    return get_rev_signal_v2(rtns, vol, s, 7, .5, .3)


def get_signal_3(rtns, vol, s):
    '''
    type B 2 weeks
    '''
    return get_b_signal(rtns, vol, s, 2, 1.6, 2.)


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
    s = r.abs().resample('W').max().rolling(13, min_periods=1).max()
    r = r.resample('W').sum()
    r = r[r.index <= dt.today()]
    w = r.abs()
    v = w[w > 0].rolling(52, min_periods=13).median()
    v[v < 5e-3] = 5e-3
    v2 = v.copy()
    v2[v2 < STOCK_VOL_FLOOR] = STOCK_VOL_FLOOR
    return r.loc[:, u.index], v.loc[:, u.index], v2.loc[:, u.index], s.loc[:, u.index]


def run_new_smx(r, v, posvol, s, capital=500):
    pos1 = get_signal_1(r, v, s)
    pos2 = get_signal_2(r, v, s)
    pos3 = get_signal_3(r, v, s)
    sig_date = pos1.index[-1]
    pos1 = (1. / posvol)[pos1 > 0].ffill(limit=3)
    pos2 = (1. / posvol)[pos2 > 0].ffill(limit=3)
    pos3 = (1. / posvol)[pos3 > 0].ffill(limit=3)
    pnl = r.mul(pos1.shift())
    pnl2 = r.mul(pos2.shift())
    pnl3 = r.mul(pos3.shift())
    rtn = r.divide(v)
    rtn = rtn.subtract(rtn.mean(axis=1), axis=0)
    s1 = rtn.rolling(6, min_periods=1).mean()
    s2 = rtn.rolling(52, min_periods=13).mean().shift(6)
    s3 = rtn.rolling(7, min_periods=1).mean()
    s4 = rtn.rolling(52, min_periods=13).mean().shift(7)
    sig = pd.concat([s1.iloc[-1], s2.iloc[-1], s3.iloc[-1], s4.iloc[-1], capital / posvol.iloc[-1]], axis=1)
    sig.columns = ['Low6', 'High6', 'Low7', 'High7', 'Position']
    sig = sig.sort_values('High6', ascending=False)
    sig = sig.dropna()
    p1 = pd.concat([pos1.iloc[-1], pnl.iloc[-1]], axis=1) * capital
    p1.columns = ['Position', 'PnL']
    p1 = p1.loc[~p1.Position.isnull()]
    p2 = pd.concat([pos2.iloc[-1], pnl2.iloc[-1]], axis=1) * capital
    p2.columns = ['Position', 'PnL']
    p2 = p2.loc[~p2.Position.isnull()]
    p3 = pd.concat([pos3.iloc[-1], pnl3.iloc[-1]], axis=1) * capital
    p3.columns = ['Position', 'PnL']
    p3 = p3.loc[~p3.Position.isnull()]
    return sig, p1, p2, p3, sig_date, pnl.sum(axis=1), pnl2.sum(axis=1), pnl3.sum(axis=1)

