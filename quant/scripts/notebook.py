'''
Created on 27 Jul 2017

@author: wayne
'''
import numpy as np
import pandas as pd
from datetime import datetime as dt
from matplotlib import pyplot as plt
from quant.lib import timeseries_utils as tu, portfolio_utils as pu, visualization_utils as vu
from quant.lib.main_utils import MODEL_PATH, load_pickle

DATABASE_NAME = 'quant'
STRATEGY_TABLE = 'strategies'


def get_now():
    return dt.today().strftime('%b %d, %Y')


def get_strategy_data(strategy_name):
    returns = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Asset Returns', 'Returns'], data_name=strategy_name)
    positions = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Positions'], data_name=strategy_name)
    signal = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Signal'], data_name=strategy_name)
    return dict(returns=returns, positions=positions, signal=signal)


def get_trading_strategy_data(strategy_name):
    returns = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Asset Returns', 'Original Returns', 'Reversal Returns'], data_name=strategy_name)
    positions = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Original Positions', 'Reversal Positions'], data_name=strategy_name)
    signal = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Signal'], data_name=strategy_name)
    return dict(returns=returns, positions=positions, signal=signal)


def get_cumulative_returns(data, lookback=None, simple_returns=False):
    r = data['returns']
    if lookback is not None:
        r = r.iloc[-lookback:]
    return r.cumsum().ffill() if simple_returns else (1. + r).cumprod().ffill() -1.
    

def get_performance_analytics(data, lookback=None, simple_returns=False):
    r = data['returns']
    if lookback is not None:
        r = r.iloc[-lookback:]
    ans = pu.get_returns_analytics(r)
    if simple_returns:
        ans['mean'] = [np.round(x, 2) for x in ans['mean']]
        ans['std'] = [np.round(x, 2) for x in ans['std']]
        ans['sharpe'] = [np.round(x, 2) for x in ans['sharpe']]
        ans['skew'] = [np.round(x, 2) for x in ans['skew']]
        ans['kurtosis'] = [np.round(x, 2) for x in ans['kurtosis']]
    else:
        ans['mean'] = ['%.1f%%' % (100. * x) for x in ans['mean']]
        ans['std'] = ['%.1f%%' % (100. * x) for x in ans['std']]
        ans['sharpe'] = [np.round(x, 2) for x in ans['sharpe']]
        ans['skew'] = [np.round(x, 2) for x in ans['skew']]
        ans['kurtosis'] = [np.round(x, 2) for x in ans['kurtosis']]
    return ans


def plot_short_returns(data, lookback, simple_returns, d1='Asset Returns', d2='Returns'):
    r = data['returns'].iloc[-lookback:]
    r3 = get_cumulative_returns(data, lookback, simple_returns)
    dd = pu.calc_drawdown(r)
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    vu.axis_area_plot(r3[d2])
    r3[d1].plot(color='black', ls='--')
    vu.use_monthly_ticks(r3)
    plt.legend(loc='best', frameon=False)
    plt.grid(ls='--')
    plt.title('Cumulative Returns', weight='bold')
    plt.subplot(222)
    vu.axis_area_plot(dd[d2], 'orange')
    dd[d1].plot(color='black', ls='--')
    vu.highlight_last_observation(dd[d2])
    vu.use_monthly_ticks(r3)
    plt.legend(loc='best', frameon=False)
    plt.grid(ls='--')
    plt.title('Drawdown', weight='bold')
    plt.subplot(223)
    rw = r.resample('W').sum()
    vu.bar_plot(rw.T, vu.get_monthly_index)
    plt.title('Weekly Returns', weight='bold')
    plt.subplot(224)
    plt.hist(r[d2], 20)
    plt.title('Returns Distribution', weight='bold')
    plt.tight_layout()


def plot_long_returns(data, simple_returns, d1='Returns', d2='Asset Returns'):
    r3 = get_cumulative_returns(data, simple_returns=simple_returns)
    dd = pu.calc_drawdown(data['returns'])
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    vu.axis_area_plot(r3[d1])
    r3[d2].plot(color='black', ls='--')
    vu.use_annual_ticks(r3)
    plt.legend(loc='best', frameon=False)
    plt.grid(ls='--')
    plt.title('Cumulative Returns', weight='bold')
    plt.subplot(122)
    vu.axis_area_plot(dd[d1], 'orange')
    dd[d2].plot(color='black', ls='--')
    vu.highlight_last_observation(dd[d1])
    vu.use_annual_ticks(r3)
    plt.legend(loc='best', frameon=False)
    plt.grid(ls='--')
    plt.title('Drawdown', weight='bold')
    plt.tight_layout()    


def plot_signals(data, lookback):
    sig = data['signal'].iloc[-lookback:]
    pos = data['positions'][sig.first_valid_index():]
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    vu.axis_area_plot(sig)
    vu.highlight_last_observation(sig)
    vu.use_monthly_ticks(sig)
    plt.grid(ls='--')
    plt.title('Signal', weight='bold')
    plt.subplot(122)
    pos.iloc[:, -1].plot()
    plt.title('Positions', weight='bold')
    plt.tight_layout()


def plot_signal_forecasting_power(data, lookback, bins):
    sig = data['signal']
    cr = tu.resample(data['returns']['Asset Returns'].cumsum(), sig).diff()
    vu.bin_plot(sig.iloc[-lookback:].shift(), cr.iloc[-lookback:], bins)
    plt.axvline(sig.iloc[-1, 0], ls='--')
    plt.xlabel('Signal')
    plt.ylabel('Forward asset returns')
    plt.title('Signal Forecasting Relationship', weight='bold')
    
    
def plot_performance_autocorrelation(data, bins):
    r = data['returns']['Asset Returns'].resample('W').sum()
    s = tu.resample(data['signal'].iloc[:, 0], r)
    x = np.sign(s) * r
    x.name = 'Past'
    y = np.sign(s) * r.shift(-1)
    y.name = 'Future'
    vu.bin_plot(x, y, bins)
    plt.title('Past vs Future performance')
    
    
def load_model_specification(simulation_name):
    filename = MODEL_PATH + '/' + simulation_name + '.model'
    load_data = load_pickle(filename)
    if load_data is None:
        return None
    else:
        selection, error_rate, _ = load_data
        return selection, error_rate
