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


def get_b_signal(r, rm, s, i=3, low=1., high=.5):
    '''
    Type B reversal signal
    '''
    s1 = rm.rolling(i, min_periods=1).mean() * np.sqrt(1. * i)
    s2 = rm.rolling(52, min_periods=13).mean().shift(i) * np.sqrt(52.)
    acc = r.cumsum().ffill()
    ax = acc.rolling(i, min_periods=1).min()
    sig = 1. * ((s1 <= -low) & (s2 >= high) & (acc == ax))
    return sig[(sig.abs() > 0) & (s <= .25)]


def get_rev_signal_v2(r, rm, s, i=3, low=1., high=.5):
    '''
    2nd Gen reversal signal
    '''
    s1 = rm.rolling(i, min_periods=1).mean()
    s2 = rm.rolling(52, min_periods=13).mean().shift(i)
    acc = r.cumsum().ffill()
    ax = acc.rolling(i, min_periods=1).min()
    sig = 1. * ((s1 <= -low) & (s2 >= high) & (acc == ax))
    return sig[(sig.abs() > 0) & (s <= .25)]


def get_signal_1(r, rm, s):
    '''
    Reversal 6 weeks
    '''
    return get_rev_signal_v2(r, rm, s, 6, .5, .3)


def get_signal_2(r, rm, s):
    '''
    reversal 7 weeks
    '''
    return get_rev_signal_v2(r, rm, s, 7, .5, .3)


def get_signal_3(r, rm, s):
    '''
    type B 2 weeks
    '''
    return get_b_signal(r, rm, s, 2, 1.7, 2.1)


def get_signal_4(r, rm, s):
    '''
    type B 6 weeks
    '''
    return get_b_signal(r, rm, s, 6, 2.1, .7)


def get_smx_data():
    u = stocks.get_ftse_smx_universe()
    r = stocks.load_google_returns(data_name='Returns', data_table=stocks.UK_STOCKS)
    r = r.loc[:, r.columns.isin(u.index)]
    return cross.get_returns(r)


def get_ftse250_data():
    u = stocks.get_ftse250_universe()
    r = stocks.load_google_returns(data_name='Returns', data_table=stocks.UK_STOCKS)
    r = r.loc[:, r.columns.isin(u.index)]
    return cross.get_returns(r)


def run_new_smx(r, rm, posvol, s, capital=500):
    pos1 = get_signal_1(r, rm, s)
    pos2 = get_signal_2(r, rm, s)
    pos3 = get_signal_3(r, rm, s)
    pos4 = get_signal_4(r, rm, s)
    sig_date = pos1.index[-1]
    pos1 = (1. / posvol)[pos1 > 0].ffill(limit=3)
    pos2 = (1. / posvol)[pos2 > 0].ffill(limit=3)
    pos3 = (1. / posvol)[pos3 > 0].ffill(limit=3)
    pos4 = (1. / posvol)[pos3 > 0].ffill(limit=3)
    pnl = r.mul(pos1.shift())
    pnl2 = r.mul(pos2.shift())
    pnl3 = r.mul(pos3.shift())
    pnl4 = r.mul(pos4.shift())
    p1 = pd.concat([pos1.iloc[-1], pnl.iloc[-1]], axis=1) * capital
    p1.columns = ['Position', 'PnL']
    p1 = p1.loc[~p1.Position.isnull()]
    p2 = pd.concat([pos2.iloc[-1], pnl2.iloc[-1]], axis=1) * capital
    p2.columns = ['Position', 'PnL']
    p2 = p2.loc[~p2.Position.isnull()]
    p3 = pd.concat([pos3.iloc[-1], pnl3.iloc[-1]], axis=1) * capital
    p3.columns = ['Position', 'PnL']
    p3 = p3.loc[~p3.Position.isnull()]
    p4 = pd.concat([pos4.iloc[-1], pnl4.iloc[-1]], axis=1) * capital
    p4.columns = ['Position', 'PnL']
    p4 = p4.loc[~p4.Position.isnull()]
    pnl = pnl.sum(axis=1)
    pnl.name = 'A6'
    pnl2 = pnl2.sum(axis=1)
    pnl2.name = 'A7'
    pnl3 = pnl3.sum(axis=1)
    pnl3.name = 'B2'
    pnl4 = pnl4.sum(axis=1)
    pnl4.name = 'B6'
    return p1, p2, p3, p4, sig_date, pnl, pnl2, pnl3, pnl4


def run_ftse250(r, rm, posvol, s, capital=500):
    pos1 = get_b_signal(r, rm, s, 2, 1.8, 1.9)
    pos2 = get_b_signal(r, rm, s, 6, 1.9, 0.6)
    sig_date = pos1.index[-1]
    pos1 = (1. / posvol)[pos1 > 0].ffill(limit=3)
    pos2 = (1. / posvol)[pos2 > 0].ffill(limit=3)
    pnl = r.mul(pos1.shift())
    pnl2 = r.mul(pos2.shift())
    p1 = pd.concat([pos1.iloc[-1], pnl.iloc[-1]], axis=1) * capital
    p1.columns = ['Position', 'PnL']
    p1 = p1.loc[~p1.Position.isnull()]
    p2 = pd.concat([pos2.iloc[-1], pnl2.iloc[-1]], axis=1) * capital
    p2.columns = ['Position', 'PnL']
    p2 = p2.loc[~p2.Position.isnull()]
    pnl = pnl.sum(axis=1)
    pnl.name = 'A6'
    pnl2 = pnl2.sum(axis=1)
    pnl2.name = 'A7'
    return p1, p2, sig_date, pnl, pnl2