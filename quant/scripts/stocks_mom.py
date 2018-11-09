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


def get_fast_signal(rtn, rm, vol, stm=3, ns=14):
    '''
    1st Gen fast signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s2 = cross.get_stock_mom(rm, 52).shift(stm)
    s3 = cross.get_stock_mom(rm, 52)
    base = 1. * (s1 >= 1.7) * (s3 >= -.1)
    ans = cross.get_step_positions(s1, s2, vol, ns, None, base, holding=0)
    return ans


def get_fast_fundamental_signal(rtn, rm, vol, score, stm=3, ns=20):
    '''
    1st Gen fast fundamental signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s3 = cross.get_stock_mom(rm, 52)
    base = 1. * (s1 >= 1.4) * (s3 >= -.4)
    ans = cross.get_step_positions(s1, score, vol, ns, None, base, holding=0)
    return ans


def get_slow_signal(rtn, rm, vol, stm=3, ns=11):
    '''
    1st Gen slow signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s2 = cross.get_stock_mom(rm, 52).shift(stm)
    s3 = cross.get_stock_mom(rm, 52)   
    base = 1. * (s1 >= 1.6) * (s3 >= -.1)
    ans = cross.get_step_positions(s2, s1, vol, ns, None, base, holding=3)
    return ans


def get_slow_fundamental_signal(rtn, rm, vol, score, stm=4, ns=10):
    '''
    1st Gen slow fundamental signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s2 = cross.get_stock_mom(rm, 52).shift(stm)
    s3 = cross.get_stock_mom(rm, 52)   
    base = 1. * (s1 >= 1.1) * (s3 >= -.5)
    ans = cross.get_step_positions(s2, score, vol, ns, None, base, holding=3)
    return ans


def get_smx_data():
    return cross.get_dataset('SMX', max_spread=.025)


def get_ftse250_data():
    return cross.get_dataset('FTSE250', max_spread=.025)


def get_aim_data():
    return cross.get_dataset('AIM', max_spread=.025)


def get_fundamentals(universe='SMX'):
    data = cross.load_financials(universe)
    data = cross.get_financials_overall_score(data)
    return data


def get_fast_bundle(r, rm, posvol, capital):
    pos = get_fast_signal(r, rm, posvol)
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


def get_fast_fundamental_bundle(r, rm, posvol, score, capital):
    pos = get_fast_fundamental_signal(r, rm, posvol, score)
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


def get_slow_bundle(r, rm, posvol, capital):
    pos = get_slow_signal(r, rm, posvol)
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


def get_slow_fundamental_bundle(r, rm, posvol, score, capital):
    pos = get_slow_fundamental_signal(r, rm, posvol, score)
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


def run_package(r, rm, posvol, score, capital=500):
    pos = []
    pnls = []
    sig_date, p1, pnl, pnl_idx = get_fast_bundle(r, rm, posvol, capital)
    _, p2, pnl2, _ = get_slow_bundle(r, rm, posvol, capital)
    _, p3, pnl3, _ = get_fast_fundamental_bundle(r, rm, posvol, score, capital)
    _, p4, pnl4, _ = get_slow_fundamental_bundle(r, rm, posvol, score, capital)
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
