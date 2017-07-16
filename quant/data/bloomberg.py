'''
Created on 22 Jun 2017

@author: wayne
'''
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta
from quant.lib.main_utils import logger
from quant.lib import timeseries_utils as tu, data_utils as du

DATABASE_NAME = 'quant'
INDEX_TABLE_NAME = 'bloomberg_index_prices'
US_ECON_TABLE_NAME = 'bloomberg_us_econ'
ACTUAL_RELEASE = 'ACTUAL_RELEASE'
ECO_RELEASE_DT = 'ECO_RELEASE_DT'
PX_LAST = 'PX_LAST'


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
    logger.info('Reading data file')
    data = read_data()
    logger.info('Loading fx rates')
    ts = format_price_data(data['FX'])
    tu.store_timeseries(ts, DATABASE_NAME, 'bloomberg_fx_rates')
    logger.info('Loading index prices')
    ts = format_price_data(data['EQ'])
    tu.store_timeseries(ts, DATABASE_NAME, 'bloomberg_index_prices')
    logger.info('Loading fund prices')
    ts = format_price_data(data['Blackrock'])
    tu.store_timeseries(ts, DATABASE_NAME, 'bloomberg_fund_prices')
    logger.info('Loading economic data')
    ts = format_econ_data(data['US'])
    for k, v in ts.iteritems():
        logger.info('Loading %s' % k)
        for kk, vv in v.iteritems():
            vv.name = k + '|' + kk
            tu.store_timeseries(vv, DATABASE_NAME, US_ECON_TABLE_NAME)


# price loader
def load_bloomberg_index_prices(tickers, start_date=None, end_date=None):
    return tu.get_timeseries(DATABASE_NAME, INDEX_TABLE_NAME, index_range=(start_date, end_date), column_list=tickers)


# Bloomberg economic data
def load_bloomberg_econ_release(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME):
    data = tu.get_timeseries(DATABASE_NAME, table_name, index_range=(start_date, end_date), column_list=[ticker + '|' + ACTUAL_RELEASE])
    release_date = tu.get_timeseries(DATABASE_NAME, table_name, index_range=(start_date, end_date), column_list=[ticker + '|' + ECO_RELEASE_DT])
    if data is not None and release_date is not None:
        release_date = release_date.drop_duplicates()
        release_date.iloc[:, 0] = [dt.strptime(str(np.int(x)), '%Y%m%d') for x in release_date.values[:, 0]]
        data = data.iloc[:, 0]
        data = data.loc[data.index.isin(release_date.index)]
        data.index = release_date.loc[data.index].values[:, 0]
        data.name = ticker
    return data


def load_bloomberg_extended_econ_release(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME):
    data = tu.get_timeseries(DATABASE_NAME, table_name, index_range=(start_date, end_date), column_list=[ticker + '|' + ACTUAL_RELEASE])
    release_date = tu.get_timeseries(DATABASE_NAME, table_name, index_range=(start_date, end_date), column_list=[ticker + '|' + ECO_RELEASE_DT])
    data2 = tu.get_timeseries(DATABASE_NAME, table_name, index_range=(start_date, end_date), column_list=[ticker + '|' + PX_LAST])
    if data is not None and release_date is not None and data2 is not None:
        release_date = release_date.drop_duplicates()
        release_date.iloc[:, 0] = [dt.strptime(str(np.int(x)), '%Y%m%d') for x in release_date.values[:, 0]]
        release_date = release_date.iloc[:, 0]
        data = data.iloc[:, 0]
        data = data.loc[data.index.isin(release_date.index)]
        data.name = ticker
        data2 = data2.iloc[:, 0]
        data2.name = ticker
        data2 = data2.loc[data2.index < data.first_valid_index()]
        data.index = release_date.loc[data.index]
        time_delta = release_date.diff().mean().days
        data2.index += timedelta(time_delta)
        data = pd.concat([data2, data], axis=0)
    return data


def load_bloomberg_econ_last(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME):
    data = tu.get_timeseries(DATABASE_NAME, table_name, index_range=(start_date, end_date), column_list=[ticker + '|' + PX_LAST])
    release_date = tu.get_timeseries(DATABASE_NAME, table_name, index_range=(start_date, end_date), column_list=[ticker + '|' + ECO_RELEASE_DT])
    if data is not None and release_date is not None:
        release_date = release_date.drop_duplicates()
        release_date.iloc[:, 0] = [dt.strptime(str(np.int(x)), '%Y%m%d') for x in release_date.values[:, 0]]
        release_date = release_date.iloc[:, 0]
        data = data.iloc[:, 0]
        data.name = ticker
        data2 = data.loc[data.index < release_date.first_valid_index()]
        data = data.loc[data.index.isin(release_date.index)]
        data.index = release_date.loc[data.index]
        time_delta = release_date.diff().mean().days
        data2.index += timedelta(time_delta)
        data = pd.concat([data2, data], axis=0)
    return data


def load_bloomberg_econ_change(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME):
    release = load_bloomberg_extended_econ_release(ticker, start_date, end_date, table_name)
    last = load_bloomberg_econ_last(ticker, start_date, end_date, table_name)
    if release is not None and last is not None:
        return release - last.shift()
    else:
        return None


def load_bloomberg_econ_time_change(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME, time_delta=252):
    release = load_bloomberg_econ_release(ticker, start_date, end_date, table_name)
    last = load_bloomberg_econ_last(ticker, start_date, end_date, table_name)
    if release is not None and last is not None:
        return release - tu.resample(last.resample('B').last().ffill().shift(time_delta), release)
    else:
        return None


def load_bloomberg_econ_revision(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME):
    release = load_bloomberg_econ_release(ticker, start_date, end_date, table_name)
    last = load_bloomberg_econ_last(ticker, start_date, end_date, table_name)
    if release is not None and last is not None:
        return (last - release).shift()
    else:
        return None


def load_bloomberg_econ_combined(ticker, start_date=None, end_date=None, table_name=US_ECON_TABLE_NAME):
    ans = []
    release = load_bloomberg_extended_econ_release(ticker, start_date, end_date, table_name)
    if release is not None:
        release.name = ticker + '|release'
        ans.append(release)
    change = load_bloomberg_econ_change(ticker, start_date, end_date, table_name)
    if change is not None:
        change.name = ticker + '|change'
        ans.append(change)
    change = load_bloomberg_econ_time_change(ticker, start_date, end_date, table_name, 252)
    if change is not None:
        change.name = ticker + '|annualchange'
        ans.append(change)
    revision = load_bloomberg_econ_revision(ticker, start_date, end_date, table_name)
    if revision is not None:
        revision.name = ticker + '|revision'
        ans.append(revision)
    return None if len(ans) == 0 else pd.concat(ans, axis=1)


def get_bloomberg_econ_list(table_name=US_ECON_TABLE_NAME):
    vals = du.get_table_column_values(DATABASE_NAME, table_name)
    return vals if vals is None else list(set([x.split('|')[0] for x in vals]))


def bloomberg_release_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, load_bloomberg_econ_release(ticker, start_date, end_date)) for ticker in tickers])


def bloomberg_extended_release_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, load_bloomberg_extended_econ_release(ticker, start_date, end_date)) for ticker in tickers])


def bloomberg_change_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, load_bloomberg_econ_change(ticker, start_date, end_date)) for ticker in tickers])


def bloomberg_annual_change_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, load_bloomberg_econ_time_change(ticker, start_date, end_date, time_delta=252)) for ticker in tickers])


def bloomberg_revision_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, load_bloomberg_econ_revision(ticker, start_date, end_date)) for ticker in tickers])


def bloomberg_combined_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, load_bloomberg_econ_combined(ticker, start_date, end_date)) for ticker in tickers])


def main():
    load_bloomberg_data()


if __name__ == '__main__':
    main()

    
    
    