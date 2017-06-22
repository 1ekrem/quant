'''
Created on 22 Jun 2017

@author: wayne
'''
import os
import logging
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


def format_econ_data(data):
    ans = {}
    i = 0
    while i < len(data.columns):
        ticker = data.columns[i].split('.')[0]
        label = data.columns[i+1].split('.')[0]
        s = pd.Series(data.values[:, i+1], index=data.values[:, i], name=ticker).dropna()
        if ticker in ans.keys():
            ans[ticker][label] = s
        else:
            ans[ticker] = dict([(label, s)])
        i += 2
    return ans


def create_tables():
    du.create_timeseries_table(DATABASE_NAME, 'bloomberg_fund_prices')
    du.create_timeseries_table(DATABASE_NAME, 'bloomberg_index_prices')
    du.create_timeseries_table(DATABASE_NAME, 'bloomberg_fx_rates')
    du.create_timeseries_table(DATABASE_NAME, 'bloomberg_us_econ')


def load_bloomberg_data():
    logging.info('Reading data file')
    data = read_data()
    logging.info('Loading fx rates')
    ts = format_price_data(data['FX'])
    tu.store_timeseries(ts, DATABASE_NAME, 'bloomberg_fx_rates')
    logging.info('Loading index prices')
    ts = format_price_data(data['EQ'])
    tu.store_timeseries(ts, DATABASE_NAME, 'bloomberg_index_prices')
    logging.info('Loading fund prices')
    ts = format_price_data(data['Blackrock'])
    tu.store_timeseries(ts, DATABASE_NAME, 'bloomberg_fund_prices')
    logging.info('Loading economic data')
    ts = format_econ_data(data['US'])
    for k, v in ts.iteritems():
        logging.info('Loading %s' % k)
        for kk, vv in v.iteritems():
            vv.name = k + '|' + kk
            tu.store_timeseries(vv, DATABASE_NAME, 'bloomberg_us_econ')


def main():
    load_bloomberg_data()


if __name__ == '__main__':
    main()

    
    
    