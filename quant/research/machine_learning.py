'''
Created on 22 Jun 2017

@author: wayne
'''
from datetime import datetime as dt
from quant.lib import timeseries_utils as tu, data_utils as du

DATABASE_NAME = 'quant'
INDEX_TABLE_NAME = 'bloomberg_index_prices'
US_ECON_TABLE_NAME = 'bloomberg_us_econ'
ACTUAL_RELEASE = 'ACTUAL_RELEASE'
PX_LAST = 'PX_LAST'


def load_bloomberg_index_prices(ticker='SPX Index'):
    return tu.get_timeseries(DATABASE_NAME, INDEX_TABLE_NAME, column_list=[ticker])


def load_bloomberg_econ_release(ticker, table_name=US_ECON_TABLE_NAME):
    data = tu.get_timeseries(DATABASE_NAME, table_name, column_list=[ticker + '|' + ACTUAL_RELEASE])
    if data is not None:
        data = data.iloc[:, 0]
        data.name = ticker
    return data


def load_bloomberg_econ_last(ticker, table_name=US_ECON_TABLE_NAME):
    data = tu.get_timeseries(DATABASE_NAME, table_name, column_list=[ticker + '|' + PX_LAST])
    if data is not None:
        data = data.iloc[:, 0]
        data.name = ticker
    return data


def load_bloomberg_econ_change(ticker, table_name=US_ECON_TABLE_NAME):
    release = load_bloomberg_econ_release(ticker, table_name)
    last = load_bloomberg_econ_last(ticker, table_name)
    if release is not None and last is not None:
        return release - last.shift()
    else:
        return None


def load_bloomberg_econ_revision(ticker, table_name=US_ECON_TABLE_NAME):
    release = load_bloomberg_econ_release(ticker, table_name)
    last = load_bloomberg_econ_last(ticker, table_name)
    if release is not None and last is not None:
        return (last - release).shift()
    else:
        return None


def get_bloomberg_econ_list(table_name=US_ECON_TABLE_NAME):
    vals = du.get_table_column_values(DATABASE_NAME, table_name)
    return vals if vals is None else list(set([x.split('|')[0] for x in vals]))


# Simulations
class USEconBoosting(object):
    '''
    Univariate forecasting
        -    Original data vs. Score
        -    Raw, change, revision
    '''
    def __init__(self, assets, start_date, end_date, frequency, sample_window, *args, **kwargs):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.sample_window = sample_window


def run_us_econ_boosting():
    sim = USEconBoosting(['SPX Index'], dt(2000, 1, 1), dt(2017, 6, 1), 'M', 24)
    return sim
    