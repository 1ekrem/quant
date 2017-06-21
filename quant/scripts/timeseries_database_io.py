'''
Created on 21 Jun 2017

@author: wayne
'''
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime as dt
from quant.lib import data_utils as du


def load_data_file():
    data = pd.read_csv(os.path.expanduser('~/TempWork/scripts/GSPC.csv'))
    data.index = pd.DatetimeIndex(data.values[:, 0])
    return data


def get_test_data():
    return pd.DataFrame([[10., 9., 8.], [10., np.nan, 2.5e-7]],
                        columns=['Wayne', 'Paul', 'James'], 
                        index=[dt(2017, 6, 1, 11, 30), dt(2017, 5, 22, 9, 0)])
    

def insert_test_data():
    data = get_test_data()
    insert_script = du.get_pandas_bulk_insert_script(data, 'Singers', 'name', 'time_index', 'value')
    delete_script = du.get_pandas_bulk_delete_script(data, 'Singers', 'name', 'time_index', 'value')
    e = du.execute_sql_script('testdb', delete_script)
    if e is not None:
        logging.warning('Failed to clear test data: ' + str(e))
    e = du.execute_sql_script('testdb', insert_script)
    if e is not None:
        logging.warning('Failed to insert test data: ' + str(e))
    