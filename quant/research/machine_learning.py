'''
Created on 22 Jun 2017

@author: wayne
'''
import numpy as np
import pandas as pd
from datetime import datetime as dt
from quant.data import bloomberg, fred, stocks
from quant.lib import timeseries_utils as tu, portfolio_utils as pu, \
    machine_learning_utils as mu
from quant.lib.main_utils import logger, load_pickle, write_pickle


DATA_MISSING_FAIL = 0.5


def _center(x):
    return x - x.median()


def get_ls_positions(signal, top=20):
    def _ls_pos(x):
        ans = np.zeros(len(x))
        v = np.sort(list(set(x[-np.isnan(x)])))
        high = 0
        low = 0
        while (high < top or low < top) and len(v) > 1:
            if high < top:
                ans[x >= v[-1]] = 1.
                high = np.sum(ans == 1)
                v = v[:-1]
            if low < top:
                ans[x <= v[0]] = -1
                low = np.sum(ans == -1)
                v = v[1:]
        if high > 0:
            ans[ans == 1] /= high
        if low > 0:
            ans[ans == -1] /= low
        return ans
    
    return signal.apply(_ls_pos, axis=1)
    


# Simulations
class StocksSim(object):
    '''
    Stocks strategy
    '''
    def __init__(self, start_date, end_date, sample_date, forecast_horizon, strategy_component,
                 universe, simulation_name, max_depth=26, model_path='', load_model=False, cross_validation=True,
                 cross_validation_buskcets=10, smart_cross_validation=True):
        self.simulation_name = simulation_name
        self.start_date = start_date
        self.end_date = end_date
        self.sample_date = sample_date
        self.forecast_horizon = forecast_horizon
        self.strategy_component = strategy_component
        self.universe = universe
        self.max_depth = max_depth
        self.model_path = model_path
        self.load_model = load_model
        self.cross_validation = cross_validation
        self.cross_validation_buckets = cross_validation_buskcets
        self.smart_cross_validation = smart_cross_validation
        assert self.forecast_horizon > 0
        self.run_sim()
    
    def run_sim(self):
        logger.info('Running simulation %s' % self.simulation_name)
        self.get_timeline()
        self.load_universe()
        self.load_stock_returns()
        self.run_simulation_sequence()
        self.calculate_returns()
        self.get_analytics()
    
    def get_timeline(self):
        logger.info('Creating time line')
        self.timeline = pd.Series([0., 0.], index=[self.start_date, self.end_date]).resample('W').last()
        self._load_start = self.timeline.index[0]
        self._load_end = self.timeline.index[-1]
    
    def load_universe(self):
        logger.info('Loading universe')
        self.u = stocks.get_universe(self.universe)

    def load_stock_returns(self):
        logger.info('Loading stock returns')
        self.stock_returns = stocks.stock_returns_loader(self.u.index)
    
    def create_estimation_data(self, depth, universe=None):
        ans = {}
        cols = ['H%d' % i for i in xrange(depth)]
        for ticker in self.u.index:
            data = self.stock_returns.get(ticker)
            if data is not None:
                res = data.Residual
                vol = data.Vol
                t = res.rolling(self.forecast_horizon).sum().shift(-self.forecast_horizon) / vol
                a = res / vol
                h = pd.concat([t] + [a.diff(i+1) for i in xrange(depth)], axis=1)
                h.columns = ['Target'] + cols
                ans[ticker] = tu.resample(h, self.timeline, carry_forward=False)
        return ans, cols

    def estimate_model(self, in_sample, out_of_sample, data, cols, model=None, use_names=None):
        '''
        When model is passed, it will be used by the strategy component
        '''
        logger.info('Running model')
        return self.strategy_component(asset_data=data, in_sample=in_sample, out_of_sample=out_of_sample,
                                       cols=cols, model=model, use_names=use_names)

    def run_without_cross_validation(self, in_sample, out_of_sample):
        data, cols = self.create_estimation_data(self.max_depth)
        return self.estimate_model(in_sample, out_of_sample, data, cols)

    def run_cross_validation(self, in_sample, out_of_sample):
        seq = mu.get_cross_validation_buckets(len(self.u.index), self.cross_validation_buckets)
        selection = None
        error_rate = 100.
        idx = 0
        error_trace = []
        for i in xrange(self.max_depth):
            logger.info('Running on depth %d of %d' % (i+1, self.max_depth + 1))
            errors = []
            data, cols = self.create_estimation_data(i+1)
            for j in xrange(self.cross_validation_buckets):
                idx += 1
                bucket = seq[j]
                validation = self.u.iloc[bucket].index
                estimation = self.u.loc[~self.u.index.isin(validation)].index
                model = self.estimate_model(in_sample, in_sample, data, cols, use_names=estimation)
                if model is not None:
                    y = []
                    x = []
                    for k, v in model.signal.iteritems():
                        if k in validation:
                            x.append(v)
                            y.append(data[k].loc[v.index, 'Target'])
                    x = pd.concat(x, axis=0).iloc[:, 0].values
                    y = pd.concat(y, axis=0).values
                    iteration_error = mu.StumpError(x, y, 0.)
                    errors.append(iteration_error)
            errors = np.mean(errors)
            error_trace.append(errors)
            logger.info('Finished, error rate %.1f%%' % (100. * errors))
            if errors < error_rate:
                error_rate = errors
                selection = i + 1
            if self.smart_cross_validation and len(error_trace) > 6:
                logger.info('Latest error rates: %s' % str(np.round(error_trace[-6:], 4)))
                if np.mean(error_trace[-6:-3]) < np.mean(error_trace[-3:]):
                    logger.info('Error rate not declining any more, search terminated.')
                    break
        if selection is None:
            return None, None, None
        else:
            data, cols = self.create_estimation_data(selection)
            return selection, error_rate, self.estimate_model(in_sample, out_of_sample, data, cols)

    def get_model_filename(self):
        return '%s/%s.stockmodel' % (self.model_path, self.simulation_name)

    def load_existing_model(self, in_sample, out_of_sample):
        logger.info('Loading model %s' % self.simulation_name)
        filename = self.get_model_filename()
        load_data = load_pickle(filename)
        if load_data is None:
            return None
        else:
            self.selection, self.error_rate, model = load_data
            data, cols = self.create_estimation_data(self.selection)
            return self.estimate_model(in_sample, out_of_sample, data, cols, model)

    def run_simulation_sequence(self):
        in_sample = self.timeline.copy()
        in_sample = in_sample.loc[in_sample.index <= self.sample_date]
        if self.forecast_horizon > 1:
            in_sample = in_sample.iloc[::self.forecast_horizon]
        in_sample = in_sample.index
        out_of_sample = self.timeline.index
        self.selection = None
        self.error_rate = None
        if self.load_model:
            comp = self.load_existing_model(in_sample, out_of_sample)
        elif self.cross_validation:
            self.selection, self.error_rate, comp = self.run_cross_validation(in_sample, out_of_sample)
            if self.error_rate is not None:
                logger.info('Error rate %.1f%% at %s' % (100. * self.error_rate, str(self.selection)))
        else:
            comp = self.run_without_cross_validation(in_sample, out_of_sample)
        if comp is None:
            logger.info('Failed to run model signal')
            self.model = None
        else:
            self.model = comp.model
            self.signal = comp.signal

    def pickle_model(self):
        filename = self.get_model_filename()
        if self.model is not None:
            logger.info('Exporting model')
            data = self.selection, self.error_rate, self.model
            write_pickle(data, filename)

    def calculate_returns(self):
        logger.info('Simulating strategy returns')
        rtns = []
        signals = []
        for k, v in self.stock_returns.iteritems():
            s = self.signal.get(k)
            if s is not None:
                x = v.Total / v.Vol.shift()
                x.name = k
                rtns.append(x)
                s = s.iloc[:, 0]
                s.name = k
                signals.append(s)
        rtns = pd.concat(rtns, axis=1)
        self.signals = pd.concat(signals, axis=1)
        s = tu.resample(self.signals, rtns, carry_forward=False).shift()
        self.ls_positions = get_ls_positions(s, top=30)
        self.positions = self.ls_positions.copy()
        self.positions[self.positions < 0.] = 0.
        start_date = self.start_date if self.start_date > self.positions.first_valid_index() else self.positions.first_valid_index()
        r = rtns.mul(self.positions).sum(axis=1)
        rls = rtns.mul(self.ls_positions).sum(axis=1)
        self.strategy_returns = pd.concat([r, rls], axis=1)
        self.strategy_returns.columns = ['Long Only', 'Long Short']
        self.strategy_returns = self.strategy_returns[start_date:]
        self.oos_strategy_returns = self.strategy_returns[self.sample_date:]

    def get_analytics(self):
        logger.info('Calculating analytics')
        self.analytics = pu.get_returns_analytics(self.strategy_returns)
        self.oos_analytics = pu.get_returns_analytics(self.oos_strategy_returns)
        

class EconSim(object):
    '''
    Pension Simulation
    '''
    def __init__(self, start_date, end_date, sample_date, data_frequency, forecast_horizon, assets,
                 asset_data_loader, inputs, input_data_loader, strategy_component, position_component,
                 simulation_name, model_path='', load_model=False, signal_model=False, data_transform_func=None,
                 default_params=None, data_missing_fail=DATA_MISSING_FAIL, simple_returns=False,
                 cross_validation=False, cross_validation_params=None, cross_validation_buckets=5,
                 smart_cross_validation=False):
        self.simulation_name = simulation_name
        self.model_path = model_path
        self.load_model = load_model
        self.signal_model = signal_model
        self.assets = assets
        self.asset_data_loader = asset_data_loader
        self.inputs = inputs
        self.input_data_loader = input_data_loader
        self.data_transform_func = data_transform_func
        self.default_params = {} if default_params is None else default_params
        self.data_missing_fail = data_missing_fail
        self.start_date = start_date
        self.end_date = end_date
        self.sample_date = sample_date
        self.data_frequency = data_frequency
        self.forecast_horizon = forecast_horizon
        self.strategy_component = strategy_component
        self.position_component = position_component
        self.simple_returns = simple_returns
        self.cross_validation = cross_validation
        self.cross_validation_params = cross_validation_params
        self.cross_validation_buckets = cross_validation_buckets
        self.smart_cross_validation = smart_cross_validation
        assert self.forecast_horizon > 0
        self.run_sim()

    def run_sim(self):
        logger.info('Running simulation %s' % self.simulation_name)
        self.get_timeline()
        self.load_asset_prices()
        self.load_input_data()
        self.run_simulation_sequence()
        self.calculate_returns()
        self.get_analytics()

    def get_timeline(self):
        logger.info('Creating time line')
        self.timeline = pu.get_timeline(self.start_date, self.end_date, self.data_frequency)
        self._load_start = self.timeline.index[0]
        self._load_end = self.timeline.index[-1]

    def get_asset_returns(self, lookback, *args, **kwargs):
        p = tu.resample(self.asset_prices, self.timeline)
        if self.simple_returns:
            asset_returns = p.diff(lookback)
        else:
            asset_returns = p.diff(lookback) / p.shift(lookback)
        return asset_returns

    def load_asset_prices(self):
        logger.info('Loading asset prices')
        self.asset_prices = self.asset_data_loader(self.assets, start_date=self._load_start, end_date=self._load_end)
        asset_returns = self.get_asset_returns(self.forecast_horizon)
        self._asset_returns = asset_returns.shift(-self.forecast_horizon)

    def load_input_data(self):
        logger.info('Loading input data')
        self.dataset = self.input_data_loader(self.inputs, start_date=self._load_start, end_date=self._load_end)

    def create_estimation_data(self, param=None):
        ans = []
        if param is None:
            param = {}
        for ticker in self.inputs:
            data = self.dataset[ticker]
            if data is not None:
                if self.data_transform_func is not None:
                    data = self.data_transform_func(data, **param)
                input = tu.resample(data, self.timeline)
                ans.append(input)
                if self.signal_model:
                    asset_returns = self.get_asset_returns(**param)
                    for c in asset_returns.columns:
                        hisrtn = np.sign(input) * asset_returns[c] 
                        hisrtn.name = c + ticker
                        ans.append(hisrtn)
        return pd.concat(ans, axis=1)

    def estimate_model(self, in_sample, out_of_sample, data, params=None, model=None):
        '''
        When model is passed, it will be used by the strategy component
        '''
        logger.info('Running model with %d variables' % np.size(data, 1))
        asset_returns = self._asset_returns.loc[in_sample.index]
        in_sample_data = pu.ignore_insufficient_series(data.loc[in_sample.index], len(in_sample) * self.data_missing_fail)
        if in_sample_data is None:
            return None
        else:
            logger.info('%d series ignored due to insufficient data' % (np.size(data, 1) - np.size(in_sample_data, 1)))
            out_of_sample_data = data.loc[out_of_sample.index, in_sample_data.columns]
            return self.strategy_component(asset_returns=asset_returns, in_sample_data=in_sample_data,
                                           out_of_sample_data=out_of_sample_data, params=params, model=model)

    def run_without_cross_validation(self, in_sample, out_of_sample):
        data = self.create_estimation_data(self.default_params)
        return self.estimate_model(in_sample, out_of_sample, data, self.default_params)

    def run_cross_validation(self, in_sample, out_of_sample):
        seq = mu.get_cross_validation_buckets(len(in_sample), self.cross_validation_buckets)
        selection = None
        error_rate = 100.
        idx = 0
        error_trace = []
        for i, param in enumerate(self.cross_validation_params):
            logger.info('Running on %s [%d of %d]' % (str(param), i+1, len(self.cross_validation_params)))
            errors = []
            validation_param = self.default_params.copy()
            validation_param.update(param)
            data = self.create_estimation_data(validation_param)
            for j in xrange(self.cross_validation_buckets):
                idx += 1
                bucket = seq[j]
                validation = in_sample.iloc[bucket]
                estimation = in_sample.loc[~in_sample.index.isin(validation.index)]
                validation_returns = self._asset_returns.loc[validation.index]
                model = self.estimate_model(estimation, validation, data, validation_param)
                if model is not None:
                    iteration_error = mu.StumpError(model.signal.iloc[:, 0], validation_returns.iloc[:, 0], 0.)
                    errors.append(iteration_error)
            errors = np.mean(errors)
            error_trace.append(errors)
            logger.info('Finished, error rate %.1f%%' % (100. * errors))
            if errors < error_rate:
                error_rate = errors
                selection = validation_param
            if self.smart_cross_validation and len(error_trace) > 6:
                logger.info('Latest error rates: %s' % str(np.round(error_trace[-6:], 4)))
                if np.mean(error_trace[-6:-3]) < np.mean(error_trace[-3:]):
                    logger.info('Error rate not declining any more, search terminated.')
                    break
        if selection is None:
            return None, None, None
        else:
            data = self.create_estimation_data(selection)
            return selection, error_rate, self.estimate_model(in_sample, out_of_sample, data, selection)

    def get_model_filename(self):
        return '%s/%s.model' % (self.model_path, self.simulation_name)

    def load_existing_model(self, in_sample, out_of_sample):
        logger.info('Loading model %s' % self.simulation_name)
        filename = self.get_model_filename()
        load_data = load_pickle(filename)
        if load_data is None:
            return None
        else:
            self.selection, self.error_rate, model = load_data
            data = self.create_estimation_data(self.selection)
            return self.estimate_model(in_sample, out_of_sample, data, self.selection, model)

    def run_simulation_sequence(self):
        in_sample = self.timeline.copy()
        in_sample = in_sample.loc[in_sample.index <= self.sample_date]
        if self.forecast_horizon > 1:
            in_sample = in_sample.iloc[::self.forecast_horizon]
        out_of_sample = self.timeline.copy()
        self.selection = None
        self.error_rate = None
        if self.load_model:
            comp = self.load_existing_model(in_sample, out_of_sample)
        elif self.cross_validation:
            self.selection, self.error_rate, comp = self.run_cross_validation(in_sample, out_of_sample)
            if self.error_rate is not None:
                logger.info('Error rate %.1f%% at %s' % (100. * self.error_rate, str(self.selection)))
        else:
            comp = self.run_without_cross_validation(in_sample, out_of_sample)
        if comp is None:
            logger.info('Failed to run model signal')
            self.model = None
        else:
            self.model = comp.model
            self.signal = comp.signal
            self.normalized_signal = comp.normalized_signal

    def pickle_model(self):
        filename = self.get_model_filename()
        if self.model is not None:
            logger.info('Exporting model')
            data = self.selection, self.error_rate, self.model
            write_pickle(data, filename)

    def calculate_returns(self):
        logger.info('Simulating strategy returns')
        p = self.asset_prices.resample('B').last().ffill()
        self.asset_returns = p.diff() if self.simple_returns else p.diff() / p.shift()
        positions = self.position_component(**{'signal': self.signal, 'normalized_signal': self.normalized_signal})
        self.positions = tu.resample(positions, self.asset_returns).fillna(0.).shift()
        start_date = self.start_date if self.start_date > self.positions.first_valid_index() else self.positions.first_valid_index()
        self.strategy_returns = self.positions.shift() * self.asset_returns
        self.strategy_returns = self.strategy_returns[start_date:]
        self.oos_strategy_returns = self.strategy_returns[self.sample_date:]
        self.asset_returns = self.asset_returns[start_date:]
        self.oos_asset_returns = self.asset_returns[self.sample_date:]

    def get_analytics(self):
        logger.info('Calculating analytics')
        self.analytics = {}
        self.oos_analytics = {}
        for asset in self.assets:
            rtn = pd.concat([self.asset_returns[asset], self.strategy_returns[asset]], axis=1)
            rtn.columns = [asset, self.simulation_name]
            self.analytics[asset] = pu.get_returns_analytics(rtn)
            self.oos_analytics[asset] = pu.get_returns_analytics(rtn[self.sample_date:])


def get_bloomberg_sim(model, input_type='release', cross_validation=False, frequency='M', long_short=False, simple_returns=False):
    econ = bloomberg.get_bloomberg_econ_list()
    if input_type == 'release':
        input_data_loader = bloomberg.bloomberg_extended_release_loader
    elif input_type == 'change':
        input_data_loader = bloomberg.bloomberg_change_loader
    elif input_type == 'annual change':
        input_data_loader = bloomberg.bloomberg_annual_change_loader
    elif input_type == 'revision':
        input_data_loader = bloomberg.bloomberg_revision_loader
    elif input_type == 'combined':
        input_data_loader = bloomberg.bloomberg_combined_loader
    if model == 'Boosting':
        strategy_component = mu.BoostingStumpComponent
    elif model == 'RandomBoosting':
        strategy_component = mu.RandomBoostingComponent
    simulation_name = '%s %s%s' % (model, input_type, ' CV' if cross_validation else '')
    position_component = pu.SimpleLongShort if long_short else pu.SimpleLongOnly
    if cross_validation:
        params = dict(cross_validation=True,
                      cross_validation_params=[{}] + [{'span': x} for x in np.arange(1, 14)],
                      cross_validation_buckets=5 if frequency=='M' else 10)
    else:
        params = {}
    sim = EconSim(assets=['SPX Index'], asset_data_loader=bloomberg.load_bloomberg_index_prices,
                  start_date=dt(2000, 1, 1), end_date=dt(2017, 6, 1), sample_date=dt(2017, 6, 1), data_frequency=frequency,
                  forecast_horizon=1, inputs=econ, input_data_loader=input_data_loader,
                  strategy_component=strategy_component, simple_returns=simple_returns, position_component=position_component,
                  simulation_name=simulation_name, data_transform_func=tu.pandas_weeks_ewma, **params)
    return sim


def get_fred_sim(model, input_type='release', cross_validation=False, frequency='M', long_short=False, simple_returns=False):
    econ = fred.get_fred_us_econ_list()
    if input_type == 'release':
        input_data_loader = fred.fred_release_loader
    elif input_type == 'change':
        input_data_loader = fred.fred_change_loader
    elif input_type == 'annual change':
        input_data_loader = fred.fred_annual_change_loader
    elif input_type == 'revision':
        input_data_loader = fred.fred_revision_loader
    elif input_type == 'combined':
        input_data_loader = fred.fred_combined_loader
    if model == 'Boosting':
        strategy_component = mu.BoostingStumpComponent
    elif model == 'RandomBoosting':
        strategy_component = mu.RandomBoostingComponent
    simulation_name = '%s %s%s' % (model, input_type, ' CV' if cross_validation else '')
    position_component = pu.SimpleLongShort if long_short else pu.SimpleLongOnly
    if cross_validation:
        params = dict(cross_validation=True,
                      cross_validation_params=[{}] + [{'span': x} for x in np.arange(1, 14)],
                      cross_validation_buckets=5 if frequency=='M' else 10)
    else:
        params = {}
    sim = EconSim(assets=['SPX Index'], asset_data_loader=bloomberg.load_bloomberg_index_prices,
                  start_date=dt(2000, 1, 1), end_date=dt(2017, 6, 1), sample_date=dt(2017, 6, 1), data_frequency=frequency,
                  forecast_horizon=1, inputs=econ, input_data_loader=input_data_loader, strategy_component=strategy_component,
                  simple_returns=simple_returns, position_component=position_component, simulation_name=simulation_name,
                  data_transform_func=tu.pandas_weeks_ewma, **params)
    return sim


def econ_run_one(model, input_type='release', cross_validation=False, oos=False, data_source='bloomberg',
                 frequency='M', long_short=False, simple_returns=False):
    if data_source == 'bloomberg':
        sim = get_bloomberg_sim(model, input_type, cross_validation, frequency, long_short, simple_returns)
    elif data_source == 'fred':
        sim = get_fred_sim(model, input_type, cross_validation, frequency, long_short, simple_returns)
    analytics = sim.oos_analytics if oos else sim.analytics
    strategy_returns = sim.oos_strategy_returns if oos else sim.strategy_returns
    asset_returns = sim.oos_asset_returns if oos else sim.asset_returns
    a = analytics['SPX Index'].iloc[1, :]
    if simple_returns:
        acc = strategy_returns.iloc[:, 0].cumsum()
    else:
        acc = (1. + strategy_returns.iloc[:, 0]).cumprod() - 1.
    acc.name = '%s (mean: %.1f%%, std: %.1f%%, sharpe: %.2f)' % (sim.simulation_name, 100. * a.loc['mean'], 100. * a.loc['std'], a.loc['sharpe'])
    a0 = analytics['SPX Index'].iloc[0, :]
    if simple_returns:
        acc0 = asset_returns.iloc[:, 0].cumsum()
    else:
        acc0 = (1. + asset_returns.iloc[:, 0]).cumprod() - 1.
    acc0.name = 'SPX Index (mean: %.1f%%, std: %.1f%%, sharpe: %.2f)' % (100. * a0.loc['mean'], 100. * a0.loc['std'], a0.loc['sharpe'])
    return acc, acc0


def run_econ_boosting(model='Boosting', oos=False, cv=False, data_source='bloomberg', frequency='M',
                      long_short=False, simple_returns=False):
    accs = []
    for input_type in ['release', 'change', 'annual change', 'revision', 'combined']:
        acc, acc0 = econ_run_one(model, input_type, cv, oos, data_source, frequency, long_short, simple_returns)
        if len(accs) == 0:
            accs.append(acc0)
        accs.append(acc)
    return pd.concat(accs, axis=1)


def run_one(model, input_type='release', data_source='bloomberg', frequency='M', long_short=False, simple_returns=False):
    if data_source == 'bloomberg':
        econ = bloomberg.get_bloomberg_econ_list()
        if input_type == 'release':
            input_data_loader = bloomberg.bloomberg_extended_release_loader
        elif input_type == 'change':
            input_data_loader = bloomberg.bloomberg_change_loader
        elif input_type == 'annual change':
            input_data_loader = bloomberg.bloomberg_annual_change_loader
        elif input_type == 'revision':
            input_data_loader = bloomberg.bloomberg_revision_loader
        elif input_type == 'combined':
            input_data_loader = bloomberg.bloomberg_combined_loader
    elif data_source == 'fred':
        econ = fred.get_fred_us_econ_list()
        if input_type == 'release':
            input_data_loader = fred.fred_release_loader
        elif input_type == 'change':
            input_data_loader = fred.fred_change_loader
        elif input_type == 'annual change':
            input_data_loader = fred.fred_annual_change_loader
        elif input_type == 'revision':
            input_data_loader = fred.fred_revision_loader
        elif input_type == 'combined':
            input_data_loader = fred.fred_combined_loader
    if model == 'Boosting':
        strategy_component = mu.BoostingStumpComponent
    elif model == 'RandomBoosting':
        strategy_component = mu.RandomBoostingComponent
    sr = -1
    ans = None
    ans0 = None
    position_component = pu.SimpleLongShort if long_short else pu.SimpleLongOnly
    for x in xrange(1, 27):
        simulation_name = '%s %s [%d]' % (model, input_type, x)
        sim = EconSim(assets=['SPX Index'], asset_data_loader=bloomberg.load_bloomberg_index_prices,
                      start_date=dt(2000, 1, 1), end_date=dt(2017, 6, 1), sample_date=dt(2017, 1, 1), data_frequency=frequency,
                      forecast_horizon=1, inputs=econ, input_data_loader=input_data_loader,
                      strategy_component=strategy_component, simple_returns=simple_returns, position_component=position_component,
                      simulation_name=simulation_name, data_transform_func=tu.pandas_weeks_ewma, default_params={'span': x})
        analytics = sim.analytics
        asset_returns = sim.asset_returns
        strategy_returns = sim.strategy_returns
        a0 = analytics['SPX Index'].iloc[0, :]
        a = analytics['SPX Index'].iloc[1, :]
        if ans0 is None:
            if simple_returns:
                acc = asset_returns.iloc[:, 0].cumsum()
                acc.name = 'SPX Index (mean: %.1f, std: %.1f, sharpe: %.2f)' % (a0.loc['mean'], a0.loc['std'], a0.loc['sharpe'])
            else:
                acc = (1. + asset_returns.iloc[:, 0]).cumprod() - 1.
                acc.name = 'SPX Index (mean: %.1f%%, std: %.1f%%, sharpe: %.2f)' % (100. * a0.loc['mean'], 100. * a0.loc['std'], a0.loc['sharpe'])
            ans0 = acc
        if a.loc['sharpe'] > sr:
            if simple_returns:
                acc = strategy_returns.iloc[:, 0].cumsum()
                acc.name = '%s (mean: %.1f, std: %.1f, sharpe: %.2f)' % (sim.simulation_name, a.loc['mean'], a.loc['std'], a.loc['sharpe'])
            else:
                acc = (1. + strategy_returns.iloc[:, 0]).cumprod() - 1.
                acc.name = '%s (mean: %.1f%%, std: %.1f%%, sharpe: %.2f)' % (sim.simulation_name, 100. * a.loc['mean'], 100. * a.loc['std'], a.loc['sharpe'])
            sr = a.loc['sharpe']
            ans = acc
    return ans, ans0


def run_boosting(model='Boosting', data_source='bloomberg', frequency='M', long_short=False, simple_returns=False):
    accs = []
    for input_type in ['release', 'change', 'annual change', 'revision', 'combined']:
        acc, acc0 = run_one(model, input_type, data_source, frequency, long_short, simple_returns)
        if len(accs) == 0:
            accs.append(acc0)
        accs.append(acc)
    return pd.concat(accs, axis=1)

