'''
Created on 25 Jun 2017

@author: wayne
'''

import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta
from quant.lib import timeseries_utils as tu
from quant.lib.main_utils import logger
from quant.lib.machine_learning_utils import get_cross_validation_buckets, StumpError


def get_timeline(start_date, end_date, frequency, sample_window=None):
    '''
    Create a timeline for simulation

    Input
    --------
    start_date       start date of simulation 
    end_date         end date of simulation
    frequency        'B', '1-5', 'M', 'MS', 'Q', 'A'
    sample_window    integer of the number of observations for sample size of estimation
    '''
    assert isinstance(start_date, dt)
    assert isinstance(end_date, dt)
    assert frequency in ['B', '1', '2', '3', '4', '5', 'M', 'MS', 'Q', 'A']
    if frequency in ['B', '1', '2', '3', '4', '5']:
        jump = 7
    elif frequency in ['M', 'MS']:
        jump = 31
    elif frequency in ['Q']:
        jump = 92
    elif frequency in ['A']:
        jump = 366
    t0 = start_date if sample_window is None else start_date - timedelta(jump * sample_window)
    base = pd.Series([0, 0], name='timeline', index=[t0, end_date])
    if frequency in ['B', '1', '2', '3', '4', '5']:
        base = base.resample('B').last()
    else:
        base = base.resample(frequency).last()
    if frequency in ['1', '2', '3', '4', '5']:
        base = base.loc[base.index.weekday.isin([np.int(frequency) - 1])]
    if frequency == 'B' and sample_window is not None:
        n = np.sum(base.index < start_date)
        base = base.iloc[n - sample_window:]
    base = base.loc[base.index <= end_date]
    return base.fillna(0.)


def ignore_insufficient_series(data, min_size):
    '''
    Get rid of series that do not have minimum size of observations
    '''
    assert isinstance(data, pd.DataFrame)
    ans = data.loc[:, (-data.isnull()).sum(axis=0) >= min_size]
    return None if ans.empty else ans

        
# portfolio components
def SimpleLongOnly(signal, *args, **kwargs):
    '''
    Simple long-only positions
    '''
    return 1. * (signal > 0)


def NormalizedSimpleLongOnly(normalized_signal, *args, **kwargs):
    '''
    Simple long-only positions using normalized signal
    '''
    return 1. * (normalized_signal > 0.5)


def SimpleLongShort(signal, *args, **kwargs):
    '''
    Simple long-only positions
    '''
    return np.sign(signal)


# performance analytics
def calc_mean(returns):
    return 52. * returns.resample('W').sum().mean(axis=0)


def calc_std(returns):
    return np.sqrt(52.) * returns.resample('W').sum().std(axis=0)


def calc_sharpe(returns):
    return calc_mean(returns) / calc_std(returns)


def calc_skew(returns):
    return returns.resample('W').sum().skew(axis=0)


def calc_kurtosis(returns):
    return returns.resample('W').sum().kurt(axis=0)


def get_returns_analytics(returns):
    ans =  pd.concat([calc_mean(returns), calc_std(returns), calc_sharpe(returns), calc_skew(returns), calc_kurtosis(returns)], axis=1)
    ans.columns = ['mean', 'std', 'sharpe', 'skew', 'kurtosis']
    return ans


def _calc_drawdown(returns):
    c = returns.cumsum().ffill().fillna(0.)
    return pd.Series([c.iloc[i] - c.ix[:i+1].max() for i in xrange(len(c))], index=returns.index, name=returns.name)


def calc_drawdown(returns):
    return _calc_drawdown(returns) if isinstance(returns, pd.Series) else returns.apply(_calc_drawdown, axis=0)


# Simulations
class TradingSim(object):
    '''
    Trading Simulation
    '''
    def __init__(self, start_date, end_date, data_frequency, assets, asset_data_loader, signal_loader,
                 simulation_name, *args, **kwargs):
        self.simulation_name = simulation_name
        self.start_date = start_date
        self.end_date = end_date
        self.data_frequency = data_frequency
        self.assets = assets
        self.asset_data_loader = asset_data_loader
        self.signal_loader = signal_loader
        self.run_sim()

    def run_sim(self):
        logger.info('Running simulation %s' % self.simulation_name)
        self.get_timeline()
        self.load_asset_prices()
        self.load_signal()
        self.run_simulation()
        self.calculate_returns()
        self.get_analytics()

    def get_timeline(self):
        logger.info('Creating time line')
        self.timeline = get_timeline(self.start_date, self.end_date, self.data_frequency)
        self._load_start = self.timeline.index[0]
        self._load_end = self.timeline.index[-1]

    def load_asset_prices(self):
        logger.info('Loading asset prices')
        self.asset_prices = self.asset_data_loader(self.assets, start_date=self._load_start, end_date=self._load_end)
        p = tu.resample(self.asset_prices, self.timeline)
        self._asset_returns = p.diff()

    def load_signal(self):
        logger.info('Loading signal')
        self.signal = self.signal_loader(start_date=self._load_start, end_date=self._load_end)

    def run_simulation(self):
        s = tu.resample(self.signal, self.timeline).iloc[:, 0]
        y = np.sign(s)
        y2 = y.copy()
        x = self._asset_returns.iloc[:, 0] * y
        y2[x < 0.] *= 2.
        y.name = 'Original'
        y2.name = 'Reversal'
        self.positions = pd.concat([y, y2], axis=1)

    def calculate_returns(self):
        logger.info('Simulating strategy returns')
        self.asset_returns = self.asset_prices.resample('B').last().ffill().diff()
        start_date = self.start_date if self.start_date > self.positions.first_valid_index() else self.positions.first_valid_index()
        strategy_returns = []
        for c in self.positions.columns:
            rtn = tu.resample(self.positions[c], self.asset_returns).shift() * self.asset_returns.iloc[:, 0]
            rtn.name = c
            strategy_returns.append(rtn)
        self.strategy_returns = pd.concat(strategy_returns, axis=1)[start_date:]
        self.asset_returns = self.asset_returns[start_date:]

    def get_analytics(self):
        logger.info('Calculating analytics')
        rtn = pd.concat([self.asset_returns, self.strategy_returns], axis=1)
        self.analytics = get_returns_analytics(rtn)
