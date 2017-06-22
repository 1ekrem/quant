'''
Created on 22 Jun 2017

@author: wayne
'''

from quant.lib import timeseries_utils as tu

DATABASE_NAME = 'quant'
INDEX_TABLE_NAME = 'bloomberg_index_prices'
US_ECON_TABLE_NAME = 'bloomberg_us_econ'
ACTUAL_RELEASE = 'ACTUAL_RELEASE'


def load_bloomberg_index_prices(ticker='SPX Index'):
    return tu.get_timeseries(DATABASE_NAME, INDEX_TABLE_NAME, column_list=[ticker])


def load_bloomberg_econ_release(ticker, table_name=US_ECON_TABLE_NAME):
    return tu.get_timeseries(DATABASE_NAME, table_name, column_list=[ticker + '|' + ACTUAL_RELEASE])