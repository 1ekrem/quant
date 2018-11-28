'''
Created on 27 Sep 2017

@author: wayne
'''
import pandas as pd
import numpy as np
from quant.lib import portfolio_utils as pu
from quant.data import stocks
from quant.research import cross, channel
from datetime import datetime as dt

STOCK_VOL_FLOOR = 0.02
MAX_SPREAD = .02


def get_fast_signal(rtn, rm, vol, stm=3, ns=4, min_fast=1.5, min_slow=-.1):
    '''
    1st Gen fast signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s2 = cross.get_stock_mom(rm, 52).shift(stm)
    s3 = cross.get_stock_mom(rm, 52)
    base = 1. * (s1 >= min_fast) * (s3 >= min_slow)
    ans = cross.get_step_positions(s1, s2, vol, ns, base, None, holding=0)
    return ans


def get_fast_fundamental_signal(rtn, rm, vol, score, stm=3, ns=4, min_fast=1.5, min_slow=-.1):
    '''
    1st Gen fast fundamental signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s3 = cross.get_stock_mom(rm, 52)
    base = 1. * (s1 >= min_fast) * (s3 >= min_slow)
    ans = cross.get_step_positions(s1, score, vol, ns, base, None, holding=0)
    return ans


def get_slow_signal(rtn, rm, vol, stm=3, ns=4, min_fast=1.5, min_slow=-.1):
    '''
    1st Gen slow signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s2 = cross.get_stock_mom(rm, 52).shift(stm)
    s3 = cross.get_stock_mom(rm, 52)   
    base = 1. * (s1 >= min_fast) * (s3 >= min_slow)
    ans = cross.get_step_positions(s2, s1, vol, ns, base, None, holding=0)
    return ans


def get_slow_fundamental_signal(rtn, rm, vol, score, stm=3, ns=4, min_fast=1.5, min_slow=-.1):
    '''
    1st Gen slow fundamental signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s2 = cross.get_stock_mom(rm, 52).shift(stm)
    s3 = cross.get_stock_mom(rm, 52)   
    base = 1. * (s1 >= min_fast) * (s3 >= min_slow)
    ans = cross.get_step_positions(s2, score, vol, ns, base, None, holding=0)
    return ans


def get_good_signal(rtn, rm, vol, stm=3, ns=6, min_fast=1.5, min_slow=.3):
    '''
    1st Gen good signal
    '''
    s1 = -1. * cross.get_stock_mom(rm, stm)
    s2 = cross.get_stock_mom(rm, 52).shift(stm)
    s3 = cross.get_stock_mom(rm, 52)
    good = s3.subtract(s3.median(axis=1), axis=0)
    base = 1. * (s1 >= min_fast) * (s3 >= min_slow) * (good >= 0.) 
    ans = cross.get_step_positions(s1, s2, vol, ns, base, None, holding=0)
    return ans


def get_smx_data():
    return cross.get_dataset('SMX', max_spread=MAX_SPREAD)


def get_ftse250_data():
    return cross.get_dataset('FTSE250', max_spread=MAX_SPREAD)


def get_aim_data():
    return cross.get_dataset('AIM', max_spread=MAX_SPREAD)


def get_fundamentals(universe='SMX'):
    data = cross.load_financials(universe)
    data = cross.get_financials_overall_score(data)
    return data

def get_channel(universe='SMX'):
    b = channel.load_breakthrough_score(universe)
    s = channel.load_support_score(universe)
    return b, s


def get_fast_bundle(r, rm, posvol, capital, stm=3, ns=4, min_fast=1.5, min_slow=-.1):
    pos = get_fast_signal(r, rm, posvol, stm=stm, ns=ns, min_fast=min_fast, min_slow=min_slow)
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


def get_fast_fundamental_bundle(r, rm, posvol, score, capital, stm=3, ns=4, min_fast=1.5, min_slow=-.1):
    pos = get_fast_fundamental_signal(r, rm, posvol, score, stm=stm, ns=ns, min_fast=min_fast, min_slow=min_slow)
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


def get_slow_bundle(r, rm, posvol, capital, stm=3, ns=4, min_fast=1.5, min_slow=-.1):
    pos = get_slow_signal(r, rm, posvol, stm=stm, ns=ns, min_fast=min_fast, min_slow=min_slow)
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


def get_slow_fundamental_bundle(r, rm, posvol, score, capital, stm=3, ns=4, min_fast=1.5, min_slow=-.1):
    pos = get_slow_fundamental_signal(r, rm, posvol, score, stm=stm, ns=ns, min_fast=min_fast, min_slow=min_slow)
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


def get_good_bundle(r, rm, posvol, capital, stm=3, ns=6, min_fast=1.5, min_slow=.3):
    pos = get_good_signal(r, rm, posvol, stm=stm, ns=ns, min_fast=min_fast, min_slow=min_slow)
    sig_date = pos.index[-1]
    pnl = r.mul(pos.shift())
    pos_idx = 1. / posvol.shift()
    pnl_idx = r.mul(pos_idx.shift())
    c = pos.sum(axis=1).mean() / pos_idx.sum(axis=1).mean()
    p1 = pd.concat([pos.iloc[-1], pnl.iloc[-1]], axis=1) * capital
    p1.columns = ['Position', 'PnL']
    p1 = p1.loc[p1.Position.abs() > 0]
    pnl = pnl.sum(axis=1)
    pnl.name = 'Good'
    pnl_idx = pnl_idx.sum(axis=1) * c
    pnl_idx.name = 'Index'
    return sig_date, p1, pnl, pnl_idx


def run_package(r, rm, posvol, score, capital=500):
    pos = []
    pnls = []
    sig_date, p1, pnl, pnl_idx = get_fast_bundle(r, rm, posvol, capital, stm=3, ns=4, min_fast=1.5, min_slow=-.1)
    _, p2, pnl2, _ = get_slow_bundle(r, rm, posvol, capital, stm=3, ns=4, min_fast=1.5, min_slow=-.1)
    _, p3, pnl3, _ = get_fast_fundamental_bundle(r, rm, posvol, score, capital, stm=3, ns=4, min_fast=1.5, min_slow=-.1)
    _, p4, pnl4, _ = get_slow_fundamental_bundle(r, rm, posvol, score, capital, stm=3, ns=4, min_fast=1.5, min_slow=-.1)
    _, p5, pnl5, _ = get_good_bundle(r, rm, posvol, capital, stm=3, ns=6, min_fast=1.5, min_slow=.3)
    pos.append(p1)
    pnls.append(pnl)
    pos.append(p2)
    pnls.append(pnl2)
    pos.append(p3)
    pnls.append(pnl3)
    pos.append(p4)
    pnls.append(pnl4)
    pos.append(p5)
    pnls.append(pnl5)
    pnls.append(pnl_idx)
    return sig_date, pos, pnls


def run_package2(r, rm, posvol, score, capital=500):
    pos = []
    pnls = []
    sig_date, p1, pnl, pnl_idx = get_fast_bundle(r, rm, posvol, capital, stm=7, ns=4, min_fast=.6, min_slow=.1)
    _, p2, pnl2, _ = get_slow_bundle(r, rm, posvol, capital, stm=7, ns=4, min_fast=.6, min_slow=.1)
    _, p3, pnl3, _ = get_fast_fundamental_bundle(r, rm, posvol, score, capital, stm=7, ns=4, min_fast=.6, min_slow=.1)
    _, p4, pnl4, _ = get_slow_fundamental_bundle(r, rm, posvol, score, capital, stm=7, ns=4, min_fast=.6, min_slow=.1)
    _, p5, pnl5, _ = get_good_bundle(r, rm, posvol, capital, stm=7, ns=9, min_fast=1.2, min_slow=.4)
    pos.append(p1)
    pnls.append(pnl)
    pos.append(p2)
    pnls.append(pnl2)
    pos.append(p3)
    pnls.append(pnl3)
    pos.append(p4)
    pnls.append(pnl4)
    pos.append(p5)
    pnls.append(pnl5)
    pnls.append(pnl_idx)
    return sig_date, pos, pnls
