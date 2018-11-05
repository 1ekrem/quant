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


def get_fast_signal(rtn, rm, vol, volume, stm=3, ns=3):
    '''
    1st Gen fast signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s2 = cross.get_stock_mom(rm, 52).shift(stm)
    s3 = cross.get_stock_mom(rm, 52)
    acc2 = rtn.cumsum()
    dd = acc2.rolling(13, min_periods=1).max() - acc2
    base = 1. * (volume >= -.6) * (dd >= .05) * (s1 >= 1.6) * (s3 >= -.5)
    ans = cross.get_step_positions(s1, s2, vol, ns, base, holding=0)
    return ans


def get_fast_fundamental_signal(rtn, rm, vol, volume, score, stm=3, ns=3):
    '''
    1st Gen fast fundamental signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s3 = cross.get_stock_mom(rm, 52)
    acc2 = rtn.cumsum()
    dd = acc2.rolling(13, min_periods=1).max() - acc2
    base = 1. * (volume >= -.6) * (dd >= .05) * (s1 >= 1.6) * (s3 >= -.5)
    ans = cross.get_step_positions(s1, score, vol, ns, base, holding=0)
    return ans


def get_slow_signal(rtn, rm, vol, volume, stm=4, ns=11):
    '''
    1st Gen slow signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s2 = cross.get_stock_mom(rm, 52).shift(stm)
    s3 = cross.get_stock_mom(rm, 52)   
    base = 1. * (volume >= -.6)
    base2 = 1. * (s1 >= 1.6) * (s3 >= .1)
    ans = cross.get_step_positions(s2, s1, vol, ns, base, base2, holding=3)
    return ans


def get_slow_fundamental_signal(rtn, rm, vol, volume, score, stm=4, ns=12):
    '''
    1st Gen slow fundamental signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s2 = cross.get_stock_mom(rm, 52).shift(stm)
    s3 = cross.get_stock_mom(rm, 52)   
    base = 1. * (volume >= -.6)
    base2 = 1. * (s1 >= 1.) * (s3 >= -.5)
    ans = cross.get_step_positions(s2, score, vol, ns, base, base2, holding=3)
    return ans


def get_smx_data():
    return cross.get_dataset('SMX')


def get_ftse250_data():
    return cross.get_dataset('FTSE250')


def get_aim_data():
    return cross.get_dataset('AIM')


def get_fundamentals(universe='SMX'):
    data = cross.load_financials(universe)
    data = cross.get_financials_overall_score(data)
    return data


def get_fast_bundle(r, rm, posvol, volume, capital):
    pos = get_fast_signal(r, rm, posvol, volume)
    sig_date = pos.index[-1]
    pnl = r.mul(pos.shift())
    pos_idx = 1. / posvol.shift()
    pnl_idx = r.mul(pos_idx.shift())
    c = pos.sum(axis=1).mean() / pos_idx.sum(axis=1).mean()
    p1 = pd.concat([pos.iloc[-1], pnl.iloc[-1]], axis=1) * capital
    p1.columns = ['Position', 'PnL']
    p1 = p1.loc[p1.Position.abs() > 0]
    pnl = pnl.sum(axis=1)
    pnl.name = 'Fast'
    pnl_idx = pnl_idx.sum(axis=1) * c
    pnl_idx.name = 'Index'
    return sig_date, p1, pnl, pnl_idx


def get_fast_fundamental_bundle(r, rm, posvol, volume, score, capital):
    pos = get_fast_fundamental_signal(r, rm, posvol, volume, score)
    sig_date = pos.index[-1]
    pnl = r.mul(pos.shift())
    pos_idx = 1. / posvol.shift()
    pnl_idx = r.mul(pos_idx.shift())
    c = pos.sum(axis=1).mean() / pos_idx.sum(axis=1).mean()
    p1 = pd.concat([pos.iloc[-1], pnl.iloc[-1]], axis=1) * capital
    p1.columns = ['Position', 'PnL']
    p1 = p1.loc[p1.Position.abs() > 0]
    pnl = pnl.sum(axis=1)
    pnl.name = 'Fast F'
    pnl_idx = pnl_idx.sum(axis=1) * c
    pnl_idx.name = 'Index'
    return sig_date, p1, pnl, pnl_idx


def get_slow_bundle(r, rm, posvol, volume, capital):
    pos = get_slow_signal(r, rm, posvol, volume)
    sig_date = pos.index[-1]
    pnl = r.mul(pos.shift())
    pos_idx = 1. / posvol.shift()
    pnl_idx = r.mul(pos_idx.shift())
    c = pos.sum(axis=1).mean() / pos_idx.sum(axis=1).mean()
    p1 = pd.concat([pos.iloc[-1], pnl.iloc[-1]], axis=1) * capital
    p1.columns = ['Position', 'PnL']
    p1 = p1.loc[p1.Position.abs() > 0]
    pnl = pnl.sum(axis=1)
    pnl.name = 'Slow'
    pnl_idx = pnl_idx.sum(axis=1) * c
    pnl_idx.name = 'Index'
    return sig_date, p1, pnl, pnl_idx    


def get_slow_fundamental_bundle(r, rm, posvol, volume, score, capital):
    pos = get_slow_fundamental_signal(r, rm, posvol, volume, score)
    sig_date = pos.index[-1]
    pnl = r.mul(pos.shift())
    pos_idx = 1. / posvol.shift()
    pnl_idx = r.mul(pos_idx.shift())
    c = pos.sum(axis=1).mean() / pos_idx.sum(axis=1).mean()
    p1 = pd.concat([pos.iloc[-1], pnl.iloc[-1]], axis=1) * capital
    p1.columns = ['Position', 'PnL']
    p1 = p1.loc[p1.Position.abs() > 0]
    pnl = pnl.sum(axis=1)
    pnl.name = 'Slow F'
    pnl_idx = pnl_idx.sum(axis=1) * c
    pnl_idx.name = 'Index'
    return sig_date, p1, pnl, pnl_idx    


def run_package(r, rm, posvol, volume, score, capital=500):
    pos = []
    pnls = []
    sig_date, p1, pnl, pnl_idx = get_fast_bundle(r, rm, posvol, volume, capital)
    _, p2, pnl2, _ = get_slow_bundle(r, rm, posvol, volume, capital)
    _, p3, pnl3, _ = get_fast_fundamental_bundle(r, rm, posvol, volume, score, capital)
    _, p4, pnl4, _ = get_slow_fundamental_bundle(r, rm, posvol, volume, score, capital)
    pos.append(p1)
    pnls.append(pnl)
    pos.append(p2)
    pnls.append(pnl2)
    pos.append(p3)
    pnls.append(pnl3)
    pos.append(p4)
    pnls.append(pnl4)
    pnls.append(pnl_idx)
    return sig_date, pos, pnls
