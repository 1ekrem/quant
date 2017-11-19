'''
Created on 18 Sep 2017

@author: wayne
'''
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta
from quant.lib import data_utils as du, timeseries_utils as tu
from quant.lib.main_utils import logger
from quant.data import quandldata


DATABASE_NAME = 'quant'
STOCKS = 'stocks'
STOCK_RETURNS = 'stock_returns'
STOCKS_DESCRIPTION = 'stocks_description'


def create_table():
    du.create_t2_timeseries_table(DATABASE_NAME, STOCKS)
    du.create_t2_timeseries_table(DATABASE_NAME, STOCK_RETURNS)
    du.create_description_table(DATABASE_NAME, STOCKS_DESCRIPTION)


def read_tickers_file(filename):
    try: 
        ans = pd.read_excel(os.path.expanduser('~/TempWork/scripts/%s.xlsx' % filename))
        ans.index = ans.ID
        return ans.Name
    except:
        return None


def import_tickers(filename, universe):
    data = read_tickers_file(filename)
    if data is not None:
        data.name = universe
        tu.store_description(data, DATABASE_NAME, STOCKS_DESCRIPTION)
    

def get_quandl_stock_id(stock_id):
    if stock_id.endswith(' LN'):
        return 'LSE/' + stock_id[:-3]
    else:
        return stock_id


def load_quandl_stock_prices(stock_id):
    series_id = get_quandl_stock_id(stock_id)
    try:
        return quandldata.load_series(series_id)
    except:
        logger.warn('Failed to load Quandl price for %s' % stock_id)
        return None


def download_stock_prices(stock_id):
    logger.info('Downloading stock prices - %s' % stock_id)
    data = load_quandl_stock_prices(stock_id)
    if data is not None:
        tu.store_timeseries(data, DATABASE_NAME, STOCKS, stock_id)


def download_stock_universe(universe):
    u = tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, [universe])
    if u is not None:
        for idx in u.index:
            download_stock_prices(idx)


def get_universe(universe):
    return tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, [universe])


def load_stock_prices(stock_id, start_date=None, end_date=None):
    return tu.get_timeseries(DATABASE_NAME, STOCKS, index_range=(start_date, end_date), data_name=stock_id)


def load_stock_index(ticker):
    if ticker == 'UKX Index':
        b = tu.get_timeseries(DATABASE_NAME, 'bloomberg_index_prices', column_list=['UKX Index'])
        q = tu.get_timeseries(DATABASE_NAME, quandldata.QUANDL_FUTURES, data_name='FTSE 100')
        q = q.Settle
        if b.last_valid_index() < q.last_valid_index():
            idx = b.last_valid_index()
            b = b[:idx].iloc[:, 0]
            q = q[q.index > idx]
            b = pd.concat([b, q], axis=0)
            b.name = 'UKX Index'
            return b
    else:
        return None


def calculate_beta_and_returns(rtns, lookback, min_obs):
    if len(rtns) >= lookback:
        rtns['Beta'] = np.nan
        rtns['Residual'] = np.nan
        rtns['Vol'] = np.nan
        rtns = rtns[['Index', 'Total', 'Beta', 'Residual', 'Vol']]
        for i in xrange(min_obs, len(rtns)+1):
            r = rtns.iloc[i-lookback:i, :2].dropna()
            if len(r) >= min_obs:
                tails = np.int(.1 * len(r))
                data = np.array(zip(r.values[:, 0], r.values[:, 1]), dtype=[('x', np.float64), ('y', np.float64)])
                data = np.sort(data, order='y')
                m = np.vstack([data['x'][tails:len(r)-tails], data['y'][tails:len(r)-tails]]).T
                m = m - np.mean(m, axis=0)
                b = np.dot(m.T, m)
                v = np.sqrt(52. * b[0, 0] / len(m))
                b = b[0, 1] / b[0, 0]
                idx = rtns.index[i-1]
                rtns.loc[idx, 'Beta'] = b
                rtns.loc[idx, 'Vol'] = v
                rtns.loc[idx, 'Residual'] = rtns.loc[idx, 'Total'] - rtns.loc[idx, 'Index'] * b
        return rtns
    else:
        return None
    

def calculate_universe_returns(universe, lookback=52, min_obs=13):
    if isinstance(universe, str):
        u = get_universe(universe)
    else:
        u = universe
    if u is not None:
        for stock_id in u.index:
            stock = load_stock_prices(stock_id)
            if stock is not None:
                logger.info('Calculating returns - %s' % stock_id)
                stock = stock['Price']
                stock = stock.ffill(limit=5)
                rtns = stock.resample('W').last()
                rtns = rtns.diff() / rtns.shift()
                vol = rtns.rolling(lookback, min_periods=min_obs).std()
                ans = pd.concat([rtns, vol], axis=1)
                ans.columns = ['Total', 'Vol']
                tu.store_timeseries(ans, DATABASE_NAME, STOCK_RETURNS, stock_id)


def stock_returns_loader(stock_ids):
    ans = {}
    for idx in stock_ids:
        r = tu.get_timeseries(DATABASE_NAME, STOCK_RETURNS, data_name=idx)
        if r is not None:
            r.loc[r.Total.abs() > .7, 'Total'] = np.nan
            ans[idx] = r
    return ans


def import_smx_tickers():
    import_tickers('smx', 'SMX Index')


def download_smx_prices():
    download_stock_universe('SMX Index')


def calculate_smx_returns():
    calculate_universe_returns('SMX Index')


def main():
    download_smx_prices()
    calculate_smx_returns()


if __name__ == '__main__':
    main()
