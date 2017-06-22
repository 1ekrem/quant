'''
Created on 22 Jun 2017

@author: wayne
'''
import os
import pandas as pd
from quant.lib import timeseries_utils as tu, data_utils as du

DATABASE_NAME = 'quant'


def read_data():
    xl = pd.ExcelFile(os.path.expanduser('~/TempWork/scripts/data.xlsx'))
    return dict([(k, xl.parse(k)) for k in xl.sheet_names])


def format_price_data(data):
    ans = []
    i = 1
    while i<len(data.columns):
        ans.append(pd.Series(data.values[:, i+1], index=data.values[:, i], name=data.columns[i+1]).dropna())
        i +=2
    return pd.concat(ans, axis=1)


def create_tables():
    du.create_timeseries_table(DATABASE_NAME, 'bloomberg_fund_prices')
    du.create_timeseries_table(DATABASE_NAME, 'bloomberg_index_prices')
    du.create_timeseries_table(DATABASE_NAME, 'bloomberg_fx_rates')
    du.create_timeseries_table(DATABASE_NAME, 'bloomberg_us_econ')
    
    
    
    