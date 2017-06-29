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
from statsmodels.sandbox.tools import cross_val


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
    for model in ['Boosting']:
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


if __name__ == '__main__':
    _ = run_univariate_econ_boosting()

