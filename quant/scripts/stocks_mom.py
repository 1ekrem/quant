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


def get_period_signal(sig):
    s = sig.fillna(0.)
    return sig[s.rolling(3).max().shift() == 0.]


def get_b_signal(r, rm, x, x2, i=3, low=1., high=.5):
    '''
    Type B reversal signal
    '''
    s1 = rm.rolling(i, min_periods=1).mean() * np.sqrt(1. * i)
    s2 = rm.rolling(52, min_periods=13).mean().shift(i) * np.sqrt(52.)
    acc = r.cumsum().ffill()
    ax = acc.rolling(i, min_periods=1).min()
    sig = 1. * ((s1 <= -low) & (s2 >= high) & (x > 0) & (x2 > 0) & (acc == ax))
    sig = sig[sig > 0]
    return get_period_signal(sig) if i > 5 else sig


def get_a_signal(r, rm, x, x2, i=3, low=1., high=.5):
    '''
    2nd Gen reversal signal
    '''
    s1 = rm.rolling(i, min_periods=1).mean()
    s2 = rm.rolling(52, min_periods=13).mean().shift(i)
    acc = r.cumsum().ffill()
    ax = acc.rolling(i, min_periods=1).min()
    sig = 1. * ((s1 <= -low) & (s2 >= high) & (x > 0) & (x2 > 0) & (acc == ax))
    sig = sig[sig > 0]
    return get_period_signal(sig) if i > 5 else sig


def get_smx_data():
    u = stocks.get_ftse_smx_universe()
    r = stocks.load_google_returns(data_name='Returns', data_table=stocks.UK_STOCKS)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rm, vol = cross.get_returns(r)
    x, x2, _, _, _, _ = cross.load_momentum_weights('SMX')
    return rtn, rm, vol, x, x2


def get_ftse250_data():
    u = stocks.get_ftse250_universe()
    r = stocks.load_google_returns(data_name='Returns', data_table=stocks.UK_STOCKS)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rm, vol = cross.get_returns(r)
    x, x2, _, _, _, _ = cross.load_momentum_weights('FTSE250')
    return rtn, rm, vol, x, x2


def run_new_smx(r, rm, posvol, x, x2, capital=500):
    pos1 = get_a_signal(r, rm, x, x2, 3, .2, 1.2)
    pos2 = get_a_signal(r, rm, x, x2, 4, .2, .9)
    pos3 = get_a_signal(r, rm, x, x2, 7, .2, .8)
    pos4 = get_b_signal(r, rm, x, x2, 3, 1.5, 2.)
    pos5 = get_b_signal(r, rm, x, x2, 4, 1.3, 1.8)
    pos6 = get_b_signal(r, rm, x, x2, 8, 1., 2.)
    sig_date = pos1.index[-1]
    pos1 = (1. / posvol)[pos1 > 0].ffill(limit=3)
    pos2 = (1. / posvol)[pos2 > 0].ffill(limit=3)
    pos3 = (1. / posvol)[pos3 > 0].ffill(limit=3)
    pos4 = (1. / posvol)[pos4 > 0].ffill(limit=3)
    pos5 = (1. / posvol)[pos5 > 0].ffill(limit=3)
    pos6 = (1. / posvol)[pos6 > 0].ffill(limit=3)
    pnl = r.mul(pos1.shift())
    pnl2 = r.mul(pos2.shift())
    pnl3 = r.mul(pos3.shift())
    pnl4 = r.mul(pos4.shift())
    pnl5 = r.mul(pos5.shift())
    pnl6 = r.mul(pos6.shift())
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
    p5 = pd.concat([pos5.iloc[-1], pnl5.iloc[-1]], axis=1) * capital
    p5.columns = ['Position', 'PnL']
    p5 = p5.loc[~p5.Position.isnull()]
    p6 = pd.concat([pos6.iloc[-1], pnl6.iloc[-1]], axis=1) * capital
    p6.columns = ['Position', 'PnL']
    p6 = p6.loc[~p6.Position.isnull()]
    pnl = pnl.sum(axis=1)
    pnl.name = 'A3'
    pnl2 = pnl2.sum(axis=1)
    pnl2.name = 'A4'
    pnl3 = pnl3.sum(axis=1)
    pnl3.name = 'A7'
    pnl4 = pnl4.sum(axis=1)
    pnl4.name = 'B3'
    pnl5 = pnl5.sum(axis=1)
    pnl5.name = 'B4'
    pnl6 = pnl6.sum(axis=1)
    pnl6.name = 'B8'
    return p1, p2, p3, p4, p5, p6, sig_date, pnl, pnl2, pnl3, pnl4, pnl5, pnl6


def run_ftse250(r, rm, posvol, x, x2, capital=500):
    pos1 = get_a_signal(r, rm, x, x2, 4, .3, .8)
    pos2 = get_a_signal(r, rm, x, x2, 6, .2, .3)
    pos3 = get_b_signal(r, rm, x, x2, 5, 1.2, 1.9)
    pos4 = get_b_signal(r, rm, x, x2, 6, 1.6, 1.6)
    sig_date = pos1.index[-1]
    pos1 = (1. / posvol)[pos1 > 0].ffill(limit=3)
    pos2 = (1. / posvol)[pos2 > 0].ffill(limit=3)
    pos3 = (1. / posvol)[pos3 > 0].ffill(limit=3)
    pos4 = (1. / posvol)[pos4 > 0].ffill(limit=3)
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
    pnl.name = 'A4'
    pnl2 = pnl2.sum(axis=1)
    pnl2.name = 'A6'
    pnl3 = pnl3.sum(axis=1)
    pnl3.name = 'B5'
    pnl4 = pnl4.sum(axis=1)
    pnl4.name = 'B6'
    return p1, p2, p3, p4, sig_date, pnl, pnl2, pnl3, pnl4