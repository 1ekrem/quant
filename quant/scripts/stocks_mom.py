'''
Created on 27 Sep 2017

@author: wayne
'''
import os
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


def main():
    run_smx()


if __name__ == '__main__':
    main()

