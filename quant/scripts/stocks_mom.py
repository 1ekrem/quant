'''
Created on 27 Sep 2017

@author: wayne
'''
import os
import pandas as pd
import numpy as np
from quant.lib import timeseries_utils as tu, portfolio_utils as pu
from quant.data import stocks
from datetime import datetime as dt

PL = (3, 14)
P17 = (6, 13)


def get_universe(universe):
    return tu.get_description(stocks.DATABASE_NAME, stocks.STOCKS_DESCRIPTION, [universe])


def load_stock_returns(universe):
    return dict([(idx, tu.get_timeseries(stocks.DATABASE_NAME, stocks.STOCK_RETURNS, data_name=idx)) for idx in universe.index])


def run_signal(stock_data, fast=7, slow=15, capital=500):
    total_returns = pd.concat([pd.Series(v.Total, name=k) for k, v in stock_data.iteritems() if v is not None], axis=1)
    volatility = pd.concat([pd.Series(v.Vol, name=k) for k, v in stock_data.iteritems() if v is not None], axis=1)
    total_returns[total_returns.abs() > .7] = np.nan
    s = total_returns.ewm(span=slow, axis=0).mean() / volatility
    f = total_returns.ewm(span=fast, axis=0).mean() / volatility
    sig = s - f
    ans = pd.concat([sig.iloc[-1], capital / volatility.iloc[-1], total_returns.iloc[-1]], axis=1)
    ans.columns = ['Signal', 'Multiplier', 'Returns']
    return ans, sig.index[-1]


def run_momentum_signal(stock_data, lag=1, lookback=26, capital=500):
    total_returns = pd.concat([pd.Series(v.Total, name=k) for k, v in stock_data.iteritems() if v is not None], axis=1)
    volatility = pd.concat([pd.Series(v.Vol, name=k) for k, v in stock_data.iteritems() if v is not None], axis=1)
    total_returns[total_returns.abs() > .7] = np.nan
    r = total_returns / volatility
    sig = r.rolling(lookback, min_periods=3).mean().shift(lag)
    ans = pd.concat([sig.iloc[-1], capital / volatility.iloc[-1], total_returns.iloc[-1]], axis=1)
    ans.columns = ['Signal', 'Multiplier', 'Returns']
    return ans, sig.index[-1]


def run_portfolio(stock_data, fast=3, slow=10, top=20):
    total_returns = pd.concat([pd.Series(v.Total, name=k) for k, v in stock_data.iteritems() if v is not None], axis=1)
    volatility = []
    for k, v in stock_data.iteritems():
        if v is not None:
            if 'Vol' in v.columns:
                tmp = v.Vol
                tmp.name = k
                volatility.append(tmp)
    volatility = pd.concat(volatility, axis=1)
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


def run_momentum_portfolio(stock_data, lag=1, lookback=26, top=20):
    total_returns = pd.concat([pd.Series(v.Total, name=k) for k, v in stock_data.iteritems() if v is not None], axis=1)
    volatility = pd.concat([pd.Series(v.Vol, name=k) for k, v in stock_data.iteritems() if v is not None], axis=1)
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


def get_stocks_data(u='SMX Index'):
    universe = get_universe(u)
    return load_stock_returns(universe)


def run_smx(stock_data=None, years=[2007, 2014, 2015, 2016, 2017]):
    if stock_data is None:
        stock_data = get_stocks_data()
    ans = {}
    for year in years:
        ans[year] = pd.DataFrame([])
    for slow in xrange(10, 53):
        for fast in xrange(2, np.min((slow - 5, 26))):
            pnl = run_portfolio(stock_data, fast=fast, slow=slow)
            for year in years:
                ans[year].loc[slow, fast] = pu.calc_sharpe(pnl[dt(year, 1, 1):])
    for year in years:
        ans[year].to_csv('~/%d.csv' % year)


def run_smx_signal(stock_data=None, capital=500.):
    if stock_data is None:
        stock_data = get_stocks_data()
    sig, sig_date = run_signal(stock_data, *P17, capital=capital)
    sig2, _ = run_signal(stock_data, *PL, capital=capital)
    sig3, _ = run_momentum_signal(stock_data, capital=capital)
    sig = pd.concat([sig, sig2, sig3], axis=1)
    sig.columns = ['Short', 'Multiplier', 'Returns', 'Long', 'M', 'R', 'Momentum', 'M2', 'R2']
    sig = sig.loc[:, ['Short', 'Long', 'Momentum', 'Multiplier', 'Returns']]
    sig = sig.sort_values('Short', ascending=False)
    return sig, sig_date


def main():
    run_smx()


if __name__ == '__main__':
    main()

    
