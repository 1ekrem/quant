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
    return np.sqrt(52.) * (rtns.rolling(mom, min_periods=1).mean() / vol)


def get_signal_1(rtns, vol):
    '''
    Momentum
    '''
    s1 = get_momentum(rtns, vol, 3)
    s2 = get_momentum(rtns, vol, 26).shift(3)
    s3 = get_momentum(rtns, vol, 52).shift(3)
    sig = s3 + s2 - s1
    return sig[(s1 > -3) & (s2 > 0) & (s3 > 0)].apply(get_top_3, axis=1)


def get_signal_2(rtns, vol):
    '''
    Recovery
    '''
    s1 = get_momentum(rtns, vol, 3)
    s2 = get_momentum(rtns, vol, 26).shift(3)
    s3 = get_momentum(rtns, vol, 52).shift(3)
    sig = s3 + s2 - s1
    return sig[(s1 > s2)].apply(get_top_3, axis=1)


def get_pnl(sig, rtns, vol):
    pos = (1. / vol)[sig > 0].ffill(limit=4)
    return get_clean_returns(rtns).mul(pos.shift()).sum(axis=1)
    

def plot(rtns, vol):
    sig = get_signal_2(rtns, vol)
    get_pnl(sig, rtns, vol).cumsum().plot()


def run_new_signal(rtns, vol, mom=13, rev=2, mom_rank=20, rev_rank=3, holding=4):
    r = rtns.copy()
    r[r.abs() > .7] = np.nan
    s1 = np.sqrt(52.) * (rtns.rolling(mom, min_periods=1).mean() / vol).shift(rev)
    s2 = np.sqrt(52.) * (rtns.rolling(rev, min_periods=1).mean() / vol)
    s = np.nan * vol
    for idx in s.index:
        tmp = s1.loc[idx].dropna().sort_values()
        if len(tmp) > mom_rank:
            tmp = tmp.iloc[-mom_rank:]
        if len(tmp) > 0:
            tmp2 = s2.loc[idx].loc[tmp.index]
            tmp2 = tmp2.dropna().sort_index()
            if len(tmp2) > rev_rank:
                tmp2 = tmp2.iloc[:rev_rank]
            if len(tmp2) > 0:
                s.loc[idx, tmp2.index] = 1.
    pos = 1. / vol
    pos = pos[s > 0]
    pos = pos.ffill(limit=holding)
    pnl = r.mul(pos.shift()).sum(axis=1)
    return {'pnl': pnl, 'pos': pos, 'rev': s2, 'mom': s1}


def run_new_portfolio(rtns, vol, mom=13, rev=2, mom_rank=20, rev_rank=3, holding=4):
    ans = run_new_signal(rtns, vol, mom, rev, mom_rank, rev_rank, holding)
    return ans['pnl']


def run_new(years=[2007, 2009, 2014, 2017]):
    r, v = get_smx_data()
    ans = {}
    for year in years:
        ans[year] = pd.DataFrame([])
    for mom in xrange(12, 53, 2):
        for rev in xrange(1, 14):
            pnl = run_new_portfolio(r, v, mom, rev)
            for year in years:
                ans[year].loc[mom, rev] = pu.calc_sharpe(pnl[dt(year, 1, 1):])
    for year in years:
        ans[year].to_csv('~/new%d.csv' % year)


def get_smx_data():
    u = stocks.get_smx_universe()
    r = stocks.load_stock_returns(data_name='Returns')
    v = stocks.load_stock_returns(data_name='Volatility')
    return r.loc[:, u.index], v.loc[:, u.index]


def run_new_smx(r, v, capital=500):
    ans = run_new_signal(r, v, 26, 3)
    ans2 = run_new_signal(r, v, 50, 3)
    sig_date = ans['rev'].index[-1]
    sig = pd.concat([ans['rev'].iloc[-1], ans['mom'].iloc[-1], ans2['mom'].iloc[-1], capital / v.iloc[-1]], axis=1)
    sig.columns = ['Reversal', 'M26', 'M50', 'Position']
    sig = sig.sort_values('M50', ascending=False)
    p = ans['pos'].iloc[-1]
    p = p[p>0]
    s = get_signal_2(r, v)
    pos = (1. / v)[s > 0].ffill(limit=4)
    pnl_x = get_clean_returns(r).mul(pos.shift()).sum(axis=1)
    return sig.dropna(), sig_date, p.index, ans['pnl'], ans2['pnl'], pnl_x 
    
