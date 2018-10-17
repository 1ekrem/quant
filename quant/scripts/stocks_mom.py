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
    sig = 1. * (s1 < -.5) * (s2 > .5)
    return sig[sig > 0]


def get_smx_data():
    return cross.get_dataset('SMX')


def get_ftse250_data():
    return cross.get_dataset('FTSE250')


def get_aim_data():
    return cross.get_dataset('AIM')


def get_a_bundle(r, rm, posvol, volume, capital, i):
    pos = get_a_signal(r, rm, i)
    v = pos.sum(axis=1).mean()
    sig_date = pos.index[-1]
    acc = r.cumsum()
    dd = acc.rolling(13, min_periods=1).max() - acc
    ltm = cross.get_stock_mom(rm, 52).shift(3)
    wl = np.sign(ltm.subtract(ltm.mean(axis=1), axis=0)).divide(posvol)
    rl = r.mul(wl.shift())
    z = rl.rolling(3, min_periods=1).mean()
    z = z.subtract(z.mean(axis=1), axis=0).divide(z.std(axis=1), axis=0)
    s1 = cross.get_stock_mom(rm, i)
    s3 = cross.get_stock_mom(rm, 52)
    pos = (1. / posvol)[pos > 0].ffill(limit=3)[(dd >= .08) & (volume >= -.6) & (z <= .1) & (s1 <=0) & (s3 >= 0)]
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
    
    
def run_package(r, rm, posvol, volume, capital=500):
    pos = []
    pnls = []
    for i in [3, 6, 9]:
        sig_date, p1, pnl, pnl_idx = get_a_bundle(r, rm, posvol, volume, capital, i)
        pos.append(p1)
        pnls.append(pnl)
    pnls.append(pnl_idx)
    return sig_date, pos, pnls
