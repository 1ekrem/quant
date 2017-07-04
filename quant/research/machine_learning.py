'''
Created on 22 Jun 2017

@author: wayne
'''
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from matplotlib import pyplot as plt
from quant.lib import timeseries_utils as tu, data_utils as du, portfolio_utils as pu, \
    machine_learning_utils as mu, visualization_utils as vu
from quant.lib.main_utils import logger


DATABASE_NAME = 'quant'
INDEX_TABLE_NAME = 'bloomberg_index_prices'
US_ECON_TABLE_NAME = 'bloomberg_us_econ'
ACTUAL_RELEASE = 'ACTUAL_RELEASE'
PX_LAST = 'PX_LAST'
DATA_MISSING_FAIL = 0.7


# price loader
def load_bloomberg_index_prices(tickers, start_date=None, end_date=None):
    return tu.get_timeseries(DATABASE_NAME, INDEX_TABLE_NAME, index_range=(start_date, end_date), column_list=tickers)


# Bloomberg economic data
def load_bloomberg_econ_release(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME):
    data = tu.get_timeseries(DATABASE_NAME, table_name, index_range=(start_date, end_date), column_list=[ticker + '|' + ACTUAL_RELEASE])
    if data is not None:
        data = data.iloc[:, 0]
        data.name = ticker
    return data


def load_bloomberg_econ_last(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME):
    data = tu.get_timeseries(DATABASE_NAME, table_name, index_range=(start_date, end_date), column_list=[ticker + '|' + PX_LAST])
    if data is not None:
        data = data.iloc[:, 0]
        data.name = ticker
    return data


def load_bloomberg_econ_change(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME):
    release = load_bloomberg_econ_release(ticker, start_date, end_date, table_name)
    last = load_bloomberg_econ_last(ticker, start_date, end_date, table_name)
    if release is not None and last is not None:
        return release - last.shift()
    else:
        return None


def load_bloomberg_econ_revision(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME):
    release = load_bloomberg_econ_release(ticker, start_date, end_date, table_name)
    last = load_bloomberg_econ_last(ticker, start_date, end_date, table_name)
    if release is not None and last is not None:
        return (last - release).shift()
    else:
        return None


def get_bloomberg_econ_list(table_name=US_ECON_TABLE_NAME):
    vals = du.get_table_column_values(DATABASE_NAME, table_name)
    return vals if vals is None else list(set([x.split('|')[0] for x in vals]))


def bloomberg_release_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, load_bloomberg_econ_release(ticker, start_date, end_date)) for ticker in tickers])


def bloomberg_change_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, load_bloomberg_econ_change(ticker, start_date, end_date)) for ticker in tickers])


def bloomberg_revision_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, load_bloomberg_econ_revision(ticker, start_date, end_date)) for ticker in tickers])


# Simulations
def univariate_run_one(input, input_type='release'):
    if input_type == 'release':
        input_data_loader = bloomberg_release_loader
    elif input_type == 'change':
        input_data_loader = bloomberg_change_loader
    elif input_type == 'revision':
        input_data_loader = bloomberg_revision_loader
    simulation_name = '%s %s' % (input, input_type)
    sim = pu.Sim(assets=['SPX Index'], asset_data_loader=load_bloomberg_index_prices,
                 start_date=dt(2002, 1, 1), end_date=dt(2017, 6, 1), data_frequency='M',
                 sample_window=24, model_frequency='Q', inputs = [input],
                 input_data_loader=input_data_loader, strategy_component=mu.BoostingStumpComponent,
                 position_component=pu.SimpleLongOnly, simulation_name=simulation_name)
    a = sim.analytics['SPX Index'].iloc[1, :]
    sharpes = sim.analytics['SPX Index']['sharpe']
    a.loc['sharpe improvement'] = sharpes.values[1] - sharpes.values[0]
    a.loc['input type'] = input_type
    a.loc['input'] = input
    return a


def run_univariate_econ_boosting():
    analytics = []
    econ = get_bloomberg_econ_list()
    for input_type in ['release', 'change', 'revision']:
        for use_scores in [False, True]:
            for input in econ:
                analytics.append(univariate_run_one(input, input_type, use_scores))
    analytics = pd.concat(analytics, axis=1).T
    analytics.to_csv(os.path.expanduser('~/TempWork/quant/univariate_boosting.csv'))
    return analytics


def multivariate_run_one(model, input_type='release', cross_validation=False):
    econ = get_bloomberg_econ_list()
    if input_type == 'release':
        input_data_loader = bloomberg_release_loader
    elif input_type == 'change':
        input_data_loader = bloomberg_change_loader
    elif input_type == 'revision':
        input_data_loader = bloomberg_revision_loader
    if model == 'Boosting':
        strategy_component = mu.BoostingStumpComponent
    elif model == 'RandomBoosting':
        strategy_component = mu.RandomBoostingComponent
    simulation_name = '%s %s%s' % (model, input_type, ' CV' if cross_validation else '')
    if cross_validation:
        params = dict(cross_validation=True, cross_validation_data_func=mu.pandas_ewma,
                      cross_validation_params=[{'span': x} for x in np.arange(2, 7)])
    else:
        params = {}
    sim = pu.Sim(assets=['SPX Index'], asset_data_loader=load_bloomberg_index_prices,
                 start_date=dt(2005,1, 1), end_date=dt(2017, 6, 1), data_frequency='M',
                 sample_window=60, model_frequency='Q', inputs=econ,
                 input_data_loader=input_data_loader, strategy_component=strategy_component,
                 position_component=pu.SimpleLongOnly, simulation_name=simulation_name, **params)
    a = sim.analytics['SPX Index'].iloc[1, :]
    acc = sim.strategy_returns.iloc[:, 0].cumsum()
    acc.name = '%s (mean: %.2f, std: %.2f, sharpe: %.2f)' % (simulation_name, a.loc['mean'], a.loc['std'], a.loc['sharpe'])
    a0 = sim.analytics['SPX Index'].iloc[0, :]
    acc0 = sim.asset_returns.iloc[:, 0].cumsum()
    acc0.name = 'SPX Index (mean: %.2f, std: %.2f, sharpe: %.2f)' % (a0.loc['mean'], a0.loc['std'], a0.loc['sharpe'])
    return acc, acc0


def run_multivariate_econ_boosting():
    accs = []
    for model in ['RandomBoosting']:
        for input_type in ['release', 'change', 'revision']:
            for cv in [True, False]:
                acc, acc0 = multivariate_run_one(model, input_type, cv)
                if len(accs) == 0:
                    accs.append(acc0)
                accs.append(acc)
    return pd.concat(accs, axis=1)


def run_signal_spline_study():
    econ = get_bloomberg_econ_list()
    input_data_loader = bloomberg_release_loader
    strategy_component = mu.RandomBoostingComponent
    simulation_name = 'boosting release'
    sim = pu.Sim(assets=['SPX Index'], asset_data_loader=load_bloomberg_index_prices,
                 start_date=dt(2005, 1, 1), end_date=dt(2017, 6, 1), data_frequency='M',
                 sample_window=60, model_frequency='Q', inputs=econ,
                 input_data_loader=input_data_loader, strategy_component=strategy_component,
                 position_component=pu.SimpleLongOnly, simulation_name=simulation_name)
    x = sim.signal.iloc[:, 0].shift()
    x.name = 'Signal'
    y = sim._asset_returns.iloc[:, 0]
    plt.figure()
    vu.bin_plot(x, y)
    x2 = sim.normalized_signal.iloc[:, 0].shift()
    x.name = 'Normalized Signal'
    plt.figure()
    vu.bin_plot(x2, y)


def cross_validation_test_case():
    econ = get_bloomberg_econ_list()
    input_data_loader = bloomberg_release_loader
    strategy_component = mu.BoostingStumpComponent
    simulation_name = 'boosting release'
    sim = pu.Sim(assets=['SPX Index'], asset_data_loader=load_bloomberg_index_prices,
                 start_date=dt(2005, 1, 1), end_date=dt(2017, 6, 1), data_frequency='M',
                 sample_window=60, model_frequency='Q', inputs=econ,
                 input_data_loader=input_data_loader, strategy_component=strategy_component,
                 position_component=pu.SimpleLongOnly, simulation_name=simulation_name,
                 cross_validation=True, cross_validation_data_func=mu.pandas_ewma,
                 cross_validation_params=[{'span': x} for x in np.arange(2, 10)])
    return sim


class EconSim(object):
    '''
    Pension Simulation
    '''
    def __init__(self, start_date, end_date, sample_date, data_frequency, assets, asset_data_loader,
                 inputs, input_data_loader, strategy_component, position_component, simulation_name,
                 cross_validation=False, cross_validation_data_func=None, cross_validation_params=None,
                 cross_validation_buckets=5):
        self.simulation_name = simulation_name
        self.assets = assets
        self.asset_data_loader = asset_data_loader
        self.inputs = inputs
        self.input_data_loader = input_data_loader
        self.start_date = start_date
        self.end_date = end_date
        self.sample_date = sample_date
        self.data_frequency = data_frequency
        self.strategy_component = strategy_component
        self.position_component = position_component
        self.cross_validation = cross_validation
        self.cross_validation_data_func = cross_validation_data_func
        self.cross_validation_params = cross_validation_params
        self.cross_validation_buckets = cross_validation_buckets
        self.run_sim()

    def run_sim(self):
        logger.info('Running simulation %s' % self.simulation_name)
        self.get_timeline()
        self.load_asset_prices()
        self.load_input_data()
        self.create_estimation_dataset()
        self.run_simulation_sequence()
        self.calculate_returns()
        self.get_analytics()

    def get_timeline(self):
        logger.info('Creating time line')
        self.timeline = pu.get_timeline(self.start_date, self.end_date, self.data_frequency)
        self._load_start = self.timeline.index[0]
        self._load_end = self.timeline.index[-1]

    def load_asset_prices(self):
        logger.info('Loading asset prices')
        self.asset_prices = self.asset_data_loader(self.assets, start_date=self._load_start, end_date=self._load_end)

    def load_input_data(self):
        logger.info('Loading input data')
        self.dataset = self.input_data_loader(self.inputs, start_date=self._load_start, end_date=self._load_end)

    def create_estimation_dataset(self):
        self._asset_returns = tu.resample(self.asset_prices, self.timeline).diff()
        data = pd.concat([tu.resample(self.dataset[ticker], self.timeline) for ticker in self.inputs], axis=1)
        self._data = data
        if self.cross_validation:
            self._validation_data = [self.cross_validation_data_func(data, **param) for param in self.cross_validation_params]

    def estimate_model(self, asset_returns, in_sample_data, out_of_sample_data):
        return self.strategy_component(asset_returns=asset_returns,
                                       in_sample_data=in_sample_data,
                                       out_of_sample_data=out_of_sample_data)

    def run_without_cross_validation(self, asset_returns, in_sample_data, out_of_sample_data):
        return self.estimate_model(asset_returns, in_sample_data, out_of_sample_data)

    def run_cross_validation(self, asset_returns, in_sample_data, out_of_sample_data):
        '''
        in_sample_data and out_of_sample_data are datasets
        '''
        seq = pu.get_cross_validation_buckets(len(asset_returns), self.cross_validation_buckets)
        selection = None
        error_rate = 1.
        for i in xrange(len(in_sample_data)):
            validation_universe = in_sample_data[i]
            errors = []
            for j in xrange(self.cross_validation_buckets):
                bucket = seq[j]
                validation_returns = asset_returns.iloc[bucket]
                estimation_returns = asset_returns.loc[~asset_returns.index.isin(validation_returns.index)]
                validation_data = validation_universe.loc[validation_returns.index]
                estimation_data = validation_universe.loc[estimation_returns.index]
                model = self.estimate_model(estimation_returns, estimation_data, validation_data)
                iteration_error = mu.StumpError(model.signal.iloc[:, 0], validation_returns.iloc[:, 0], 0.)
                errors.append(iteration_error)
            errors = np.mean(errors)
            if errors < error_rate:
                error_rate = errors
                selection = i
        return selection, self.estimate_model(asset_returns, in_sample_data[selection], out_of_sample_data[selection])
        
    def run_simulation_sequence(self):
        in_sample = self.timeline.copy()
        in_sample = in_sample.loc[in_sample.index <= self.sample_date]
        in_sample_data = pu.ignore_insufficient_series(self._data.loc[in_sample.index], 20)
        asset_returns = self._asset_returns.loc[in_sample.index]
        if in_sample_data is None:
            logger.info('Insufficient Data')
        else:
            logger.info('Running model')
            out_of_sample = self.timeline.copy()
            out_of_sample_data = self._data.loc[out_of_sample.index, in_sample_data.columns]
            if self.cross_validation:
                in_sample_dataset = [x.loc[in_sample_data.index, in_sample_data.columns] for x in self._validation_data]
                out_of_sample_dataset = [x.loc[out_of_sample_data.index, out_of_sample_data.columns] for x in self._validation_data]                    
                selection, comp = self.run_cross_validation(asset_returns, in_sample_dataset, out_of_sample_dataset)
                logger.info('Cross validation found solution at %s' % str(self.cross_validation_params[selection]))
            else:
                comp = self.run_without_cross_validation(asset_returns, in_sample_data, out_of_sample_data)
            self.model = comp.model
            self.signal = comp.signal
            self.normalized_signal = comp.normalized_signal

    def calculate_returns(self):
        logger.info('Simulating strategy returns')
        self.asset_returns = self.asset_prices.resample('B').last().ffill().diff()
        self.positions = self.position_component(**{'signal': self.signal, 'normalized_signal': self.normalized_signal})
        start_date = self.start_date if self.start_date > self.positions.first_valid_index() else self.positions.first_valid_index()
        self.strategy_returns = tu.resample(self.positions, self.asset_returns).shift() * self.asset_returns
        self.strategy_returns = self.strategy_returns[start_date:]
        self.asset_returns = self.asset_returns[start_date:]

    def get_analytics(self):
        logger.info('Calculating analytics')
        self.analytics = {}
        for asset in self.assets:
            rtn = pd.concat([self.asset_returns[asset], self.strategy_returns[asset]], axis=1)
            rtn.columns = [asset, self.simulation_name]
            self.analytics[asset] = pu.get_returns_analytics(rtn)


def econ_run_one(model, input_type='release', cross_validation=False):
    econ = get_bloomberg_econ_list()
    if input_type == 'release':
        input_data_loader = bloomberg_release_loader
    elif input_type == 'change':
        input_data_loader = bloomberg_change_loader
    elif input_type == 'revision':
        input_data_loader = bloomberg_revision_loader
    if model == 'Boosting':
        strategy_component = mu.BoostingStumpComponent
    elif model == 'RandomBoosting':
        strategy_component = mu.RandomBoostingComponent
    simulation_name = '%s %s%s' % (model, input_type, ' CV' if cross_validation else '')
    if cross_validation:
        params = dict(cross_validation=True, cross_validation_data_func=mu.pandas_ewma,
                      cross_validation_params=[{'span': x} for x in np.arange(2, 13)],
                      cross_validation_buckets=5)
    else:
        params = {}
    sim = EconSim(assets=['SPX Index'], asset_data_loader=load_bloomberg_index_prices,
                 start_date=dt(2000,1, 1), end_date=dt(2017, 6, 1), sample_date=dt(2010, 1,1), data_frequency='M',
                 inputs=econ, input_data_loader=input_data_loader, strategy_component=strategy_component,
                 position_component=pu.SimpleLongOnly, simulation_name=simulation_name, **params)
    a = sim.analytics['SPX Index'].iloc[1, :]
    acc = sim.strategy_returns.iloc[:, 0].cumsum()
    acc.name = '%s (mean: %.2f, std: %.2f, sharpe: %.2f)' % (simulation_name, a.loc['mean'], a.loc['std'], a.loc['sharpe'])
    a0 = sim.analytics['SPX Index'].iloc[0, :]
    acc0 = sim.asset_returns.iloc[:, 0].cumsum()
    acc0.name = 'SPX Index (mean: %.2f, std: %.2f, sharpe: %.2f)' % (a0.loc['mean'], a0.loc['std'], a0.loc['sharpe'])
    return acc, acc0


def run_econ_boosting():
    accs = []
    for model in ['Boosting']:
        for input_type in ['release', 'change', 'revision']:
            for cv in [True, False]:
                acc, acc0 = econ_run_one(model, input_type, cv)
                if len(accs) == 0:
                    accs.append(acc0)
                accs.append(acc)
    return pd.concat(accs, axis=1)


if __name__ == '__main__':
    _ = run_univariate_econ_boosting()

