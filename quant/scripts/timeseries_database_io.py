'''
Created on 21 Jun 2017

@author: wayne
'''
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from quant.lib.main_utils import logger
from quant.lib import data_utils as du, timeseries_utils as tu


def load_data_file():
    data = pd.read_csv(os.path.expanduser('~/TempWork/scripts/GSPC.csv'))
    data.index = pd.DatetimeIndex(data.values[:, 0])
    return data.iloc[:, 1:]


def get_test_data():
    return pd.DataFrame([[10., 9., 8.], [10., np.nan, 2.5e-6]],
                        columns=['Wayne', 'Paul', 'James'], 
                        index=[dt(2017, 6, 1, 11, 30), dt(2017, 5, 22, 9, 0)])
    

def create_test_table():
    du.create_timeseries_table('testdb', 'ts')


def insert_test_data():
    data = get_test_data()
    insert_script = du.get_pandas_bulk_insert_script(data, 'Singers', 'name', 'time_index', 'value')
    delete_script = du.get_pandas_bulk_delete_script(data, 'Singers', 'name', 'time_index')
    e = du.execute_sql_input_script('testdb', delete_script)
    if e is not None:
        logger.warning('Failed to clear test data: ' + str(e))
    e = du.execute_sql_input_script('testdb', insert_script)
    if e is not None:
        logger.warning('Failed to insert test data: ' + str(e))


def new_insert_test_data():
    data = get_test_data()
    du.pandas_bulk_insert(data, 'testdb', 'Singers', 'name', 'time_index', 'value')


def read_test_data():
    print(du.pandas_read('testdb', 'Singers', 'name', 'time_index', 'value', None, None))
    print(du.pandas_read('testdb', 'Singers', 'name', 'time_index', 'value', (dt(2017, 6, 1), None), None))
    print(du.pandas_read('testdb', 'Singers', 'name', 'time_index', 'value', (dt(2017, 7, 1), None), None))
    print(du.pandas_read('testdb', 'Singers', 'name', 'time_index', 'value', (None, dt(2017, 6, 1)), None))
    print(du.pandas_read('testdb', 'Singers', 'name', 'time_index', 'value', (dt(2016, 5, 1), dt(2017, 6, 1)), None))
    print(du.pandas_read('testdb', 'Singers', 'name', 'time_index', 'value', None, ['Wayne', 'Paul']))
    print(du.pandas_read('testdb', 'Singers', 'name', 'time_index', 'value', (None, dt(2017, 6, 1)), ['Wayne', 'Paul']))


def insert_yahoo_data():
    data = load_data_file()
    tu.store_timeseries(data, 'testdb', 'ts')


def read_yahoo_data():
    print(tu.get_timeseries('testdb', 'ts'))
    print(tu.get_timeseries('testdb', 'ts', (dt(2017,6,1), None)))
    print(tu.get_timeseries('testdb', 'ts', (None, dt(2017,6,1)), ['Adj Close', 'Volume']))
    
    
