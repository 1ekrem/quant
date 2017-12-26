'''
Created on 27 Sep 2017

@author: wayne
'''
import pandas as pd
import numpy as np
from quant.lib import portfolio_utils as pu
from quant.data import stocks
from datetime import datetime as dt

PL = (4, 15)
P17 = (9, 45)


def run_signal(rtns, vol, fast=7, slow=15, capital=500):
    total_returns = rtns.copy()
    volatility = vol.copy()
    total_returns[total_returns.abs() > .7] = np.nan
    s = total_returns.ewm(span=slow, axis=0).mean() / volatility
    f = total_returns.ewm(span=fast, axis=0).mean() / volatility
    sig = s - f
    ans = pd.concat([sig.iloc[-1], capital / volatility.iloc[-1], total_returns.iloc[-1]], axis=1)
    ans.columns = ['Signal', 'Multiplier', 'Returns']
    return ans, sig.index[-1]


def run_momentum_signal(rtns, vol, lag=1, lookback=26, capital=500):
    total_returns = rtns.copy()
    volatility = vol.copy()
    total_returns[total_returns.abs() > .7] = np.nan
    r = total_returns / volatility
    sig = r.rolling(lookback, min_periods=3).mean().shift(lag)
    ans = pd.concat([sig.iloc[-1], capital / volatility.iloc[-1], total_returns.iloc[-1]], axis=1)
    ans.columns = ['Signal', 'Multiplier', 'Returns']
    return ans, sig.index[-1]


def run_portfolio(r, v, fast=3, slow=10, top=20):
    total_returns = r.copy()
    volatility = v.copy()
    total_returns[total_returns.abs() > .7] = np.nan
    s = total_returns.ewm(span=slow, axis=0).mean() / volatility
    f = total_returns.ewm(span=fast, axis=0).mean() / volatility
    sig = s - f
    
    def get_top(x):
        xx = np.sort(x.dropna().values)
        if len(xx) > 0:
            cutoff = xx[-top] if len(xx) >= top else xx[0]
        else:
            cutoff = 0.
        return 1. * (x >= cutoff)
    
    def normalize(x):
        xx = x.sum()
        return x / xx if xx > 0 else x * np.nan

    pos = sig.apply(get_top, axis=1)
    weight = volatility.copy()
    weight[weight <= 0.05] = 0.05
    pos *= 1 / weight
    pos = pos.apply(normalize, axis=1)
    rtn = pos.shift() * total_returns
    pnl = rtn.sum(axis=1)
    return pnl


def run_momentum_portfolio(rtns, vol, lag=1, lookback=26, top=20):
    total_returns = rtns.copy()
    volatility = vol.copy()
    total_returns[total_returns.abs() > .7] = np.nan
    r = total_returns / volatility
    sig = r.rolling(lookback, min_periods=3).mean().shift(lag)
    
    def get_top(x):
        xx = np.sort(x.dropna().values)
        if len(xx) > 0:
            cutoff = xx[-top] if len(xx) >= top else xx[0]
            return 1. * (x >= cutoff)
        else:
            return x
    
    def normalize(x):
        xx = x.sum()
        return x / xx if xx > 0 else x * np.nan

    pos = sig.apply(get_top, axis=1)
    weight = volatility.copy()
    weight[weight <= 0.05] = 0.05
    pos *= 1 / weight
    pos = pos.apply(normalize, axis=1)
    rtn = pos.shift() * total_returns
    pnl = rtn.sum(axis=1)
    return pnl


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


def run_smx(years=[2007, 2014, 2015, 2016, 2017]):
    r, v = get_smx_data()
    ans = {}
    for year in years:
        ans[year] = pd.DataFrame([])
    for slow in xrange(10, 53):
        for fast in xrange(2, np.min((slow - 5, 26))):
            pnl = run_portfolio(r, v, fast=fast, slow=slow)
            for year in years:
                ans[year].loc[slow, fast] = pu.calc_sharpe(pnl[dt(year, 1, 1):])
    for year in years:
        ans[year].to_csv('~/%d.csv' % year)


def run_smx_signal(r, v, capital=500.):
    sig, sig_date = run_signal(r, v, *P17, capital=capital)
    sig2, _ = run_signal(r, v, *PL, capital=capital)
    sig3, _ = run_momentum_signal(r, v, capital=capital)
    sig = pd.concat([sig, sig2, sig3], axis=1)
    sig.columns = ['Short', 'Multiplier', 'Returns', 'Long', 'M', 'R', 'Momentum', 'M2', 'R2']
    sig = sig.loc[:, ['Short', 'Long', 'Momentum', 'Multiplier', 'Returns']]
    sig = sig.sort_values('Short', ascending=False)
    return sig, sig_date


def run_new_smx(r, v, capital=500):
    ans = run_new_signal(r, v, 26, 3)
    ans2 = run_new_signal(r, v, 50, 3)
    sig_date = ans['rev'].index[-1]
    sig = pd.concat([ans['rev'].iloc[-1], ans['mom'].iloc[-1], ans2['mom'].iloc[-1], capital / v.iloc[-1]], axis=1)
    sig.columns = ['Reversal', 'M26', 'M50', 'Position']
    sig = sig.sort_values('M50', ascending=False)
    p = ans['pos'].iloc[-1]
    p = p[p>0]
    return sig.dropna(), sig_date, p.index, ans['pnl'], ans2['pnl'] 
    

def main():
    run_smx()


if __name__ == '__main__':
    main()

