'''
Created on 22 Jun 2017

@author: wayne
'''
import os
import pandas as pd
from datetime import datetime as dt
from quant.lib import timeseries_utils as tu, data_utils as du, portfolio_utils as pu, machine_learning_utils as mu


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
def univariate_run_one(input, input_type='release', use_scores=False):
    if input_type == 'release':
        input_data_loader = bloomberg_release_loader
    elif input_type == 'change':
        input_data_loader = bloomberg_change_loader
    elif input_type == 'revision':
        input_data_loader = bloomberg_revision_loader
    simulation_name = '%s %s %s' % (input, input_type, 'scores' if use_scores else 'original')
    sim = pu.Sim(assets=['SPX Index'], asset_data_loader=load_bloomberg_index_prices,
                 start_date=dt(2002, 1, 1), end_date=dt(2017, 6, 1), data_frequency='M',
                 sample_window=24, model_frequency='Q', inputs = [input],
                 input_data_loader=input_data_loader, strategy_component=mu.BoostingStumpComponent,
                 use_scores = use_scores, position_component=pu.NormalizedSimpleLongOnly,
                 simulation_name=simulation_name)
    a = sim.analytics['SPX Index'].iloc[1, :]
    sharpes = sim.analytics['SPX Index']['sharpe']
    a.loc['sharpe improvement'] = sharpes.values[1] - sharpes.values[0]
    a.loc['input type'] = input_type
    a.loc['input'] = input
    a.loc['scores'] = use_scores
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


if __name__ == '__main__':
    _ = run_univariate_econ_boosting()

