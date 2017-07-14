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
    if frequency in ['1', '2data', '3', '4', '5']:
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


# performance analytics
def calc_mean(returns):
    return 52. * returns.resample('W').sum().mean(axis=0)


def calc_std(returns):
    return np.sqrt(52.) * returns.resample('W').sum().std(axis=0)


def calc_sharpe(returns):
    return calc_mean(returns) / calc_std(returns)


def get_returns_analytics(returns):
    ans =  pd.concat([calc_mean(returns), calc_std(returns), calc_sharpe(returns)], axis=1)
    ans.columns = ['mean', 'std', 'sharpe']
    return ans


# Simulations
class Sim(object):
    '''
    Simulation

    Input
    --------
    start_date
    end_date
    data_frequency
    model_frequency
    sample_window
    assets
    asset_data_loader
    inputs
    input_data_loader
    position_component
    simulation_name
    data_missing_pass

    '''
    def __init__(self, start_date, end_date, data_frequency, model_frequency, sample_window,
                 assets, asset_data_loader, inputs, input_data_loader, strategy_component,
                 position_component, simulation_name, data_missing_pass=0.7,
                 cross_validation=False, cross_validation_data_func=None, cross_validation_params=None,
                 cross_validation_buckets=5):
        self.simulation_name = simulation_name
        self.assets = assets
        self.asset_data_loader = asset_data_loader
        self.inputs = inputs
        self.input_data_loader = input_data_loader
        self.start_date = start_date
        self.end_date = end_date
        self.data_frequency = data_frequency
        self.sample_window = sample_window
        self.model_frequency = model_frequency
        self.strategy_component = strategy_component
        self.position_component = position_component
        self.data_missing_pass = data_missing_pass
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
        self.timeline = get_timeline(self.start_date, self.end_date, self.data_frequency, self.sample_window)
        self.model_timeline = self.timeline.resample(self.model_frequency).last()
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
        seq = get_cross_validation_buckets(len(asset_returns), self.cross_validation_buckets)
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
                iteration_error = StumpError(model.signal.iloc[:, 0], validation_returns.iloc[:, 0], 0.)
                errors.append(iteration_error)
            errors = np.mean(errors)
            if errors < error_rate:
                error_rate = errors
                selection = i
        return selection, self.estimate_model(asset_returns, in_sample_data[selection], out_of_sample_data[selection])
        
    def run_simulation_sequence(self):
        self.models = []
        self.signal = []
        self.normalized_signal = []
        for idx, model_time in enumerate(self.model_timeline.index):
            in_sample = self.timeline.copy()
            in_sample = in_sample.loc[in_sample.index <= model_time]
            in_sample = in_sample.iloc[-self.sample_window:]
            in_sample_data = ignore_insufficient_series(self._data.loc[in_sample.index],
                                                        self.sample_window * self.data_missing_pass)
            asset_returns = self._asset_returns.loc[in_sample.index]
            if in_sample_data is None:
                logger.info('Insufficient Data - ignored %s' % model_time.strftime('%Y-%m-%d'))
            else:
                logger.info('Running model at %s' % model_time.strftime('%Y-%m-%d'))
                out_of_sample = self.timeline.copy()
                out_of_sample = out_of_sample.loc[out_of_sample.index > model_time]
                if idx < len(self.model_timeline) - 1:
                    next_date = self.model_timeline.index[idx + 1]
                    out_of_sample = out_of_sample.loc[out_of_sample.index <= next_date]
                out_of_sample_data = self._data.loc[out_of_sample.index, in_sample_data.columns]
                
                if self.cross_validation:
                    in_sample_dataset = [x.loc[in_sample_data.index, in_sample_data.columns] for x in self._validation_data]
                    out_of_sample_dataset = [x.loc[out_of_sample_data.index, out_of_sample_data.columns] for x in self._validation_data]                    
                    selection, comp = self.run_cross_validation(asset_returns, in_sample_dataset, out_of_sample_dataset)
                    logger.info('Cross validation found solution at %s' % str(self.cross_validation_params[selection]))
                else:
                    comp = self.run_without_cross_validation(asset_returns, in_sample_data, out_of_sample_data)
                self.models.append((model_time, comp.model))
                self.signal.append(comp.signal)
                self.normalized_signal.append(comp.normalized_signal)
        self.signal = pd.concat(self.signal, axis=0)
        self.normalized_signal = pd.concat(self.normalized_signal, axis=0)

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
            self.analytics[asset] = get_returns_analytics(rtn)
