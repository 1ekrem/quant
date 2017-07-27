'''
Created on 27 Jul 2017

@author: wayne
'''
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt
from quant.lib import timeseries_utils as tu, portfolio_utils as pu, visualization_utils as vu

DATABASE_NAME = 'quant'
STRATEGY_TABLE = 'strategies'


def print_now():
    print('Information updated at %s' % dt.today().strftime('%b %d, %Y'))


def get_strategy_data(strategy_name):
    returns = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Asset Returns', 'Returns'], data_name=strategy_name)
    positions = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Positions'], data_name=strategy_name)
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
        
    else:
        ans['mean'] = ['%.1f%%' % (100. * x) for x in ans['mean']]
        ans['std'] = ['%.1f%%' % (100. * x) for x in ans['std']]
        ans['sharpe'] = [np.round(x, 2) for x in ans['sharpe']]
    return ans


def plot_short_returns(data, lookback, simple_returns):
    r = data['returns'].iloc[-lookback:]
    r3 = get_cumulative_returns(data, lookback, simple_returns)
    dd = pu.calc_drawdown(r)
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    vu.axis_area_plot(r3['Returns'])
    r3['Asset Returns'].plot(color='black', ls='--')
    vu.use_monthly_ticks(r3)
    plt.legend(loc='best', frameon=False)
    plt.grid(ls='--')
    plt.title('Cumulative Returns', weight='bold')
    plt.subplot(222)
    vu.axis_area_plot(dd['Returns'], 'orange')
    dd['Asset Returns'].plot(color='black', ls='--')
    vu.highlight_last_observation(dd['Returns'])
    vu.use_monthly_ticks(r3)
    plt.legend(loc='best', frameon=False)
    plt.grid(ls='--')
    plt.title('Drawdown', weight='bold')
    plt.subplot(223)
    rw = r.resample('W').sum()
    vu.bar_plot(rw.T, vu.get_monthly_index)
    plt.title('Weekly Returns')
    plt.tight_layout()


def plot_long_returns(data, simple_returns):
    r3 = get_cumulative_returns(data, simple_returns=simple_returns)
    dd = pu.calc_drawdown(data['returns'])
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    vu.axis_area_plot(r3['Returns'])
    r3['Asset Returns'].plot(color='black', ls='--')
    vu.use_annual_ticks(r3)
    plt.legend(loc='best', frameon=False)
    plt.grid(ls='--')
    plt.title('Cumulative Returns', weight='bold')
    plt.subplot(122)
    vu.axis_area_plot(dd['Returns'], 'orange')
    dd['Asset Returns'].plot(color='black', ls='--')
    vu.highlight_last_observation(dd['Returns'])
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
    pos.iloc[:, 0].plot()
    plt.title('Positions', weight='bold')
    plt.tight_layout()
