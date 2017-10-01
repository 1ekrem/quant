'''
Created on 18 Sep 2017

@author: wayne
'''
import os
import numpy as np
import pandas as pd
import fix_yahoo_finance as yf
from datetime import datetime as dt, timedelta
from quant.lib import data_utils as du, timeseries_utils as tu
from quant.lib.main_utils import logger
from pandas_datareader import data as pdr
from quant.data import quandldata
yf.pdr_override()

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


def get_yahoo_stock_id(stock_id):
    if stock_id.endswith(' LN'):
        return stock_id[:-3] + '.L'
    else:
        return stock_id


def load_yahoo_stock_prices(stock_id, start_date=dt(2010,1,1), end_date=dt.today()):
    yahoo_id = get_yahoo_stock_id(stock_id)
    try:
        return pdr.get_data_yahoo(yahoo_id, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    except:
        logger.warn('Failed to load Yahoo price for %s' % stock_id)
        return None
    

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
    # data = load_quandl_stock_prices(stock_id)
    data = load_yahoo_stock_prices(stock_id, dt(2000,1,1), dt.today())
    if data is not None:
        try:
            if len(data) > 1:
                c = data / data.shift()
                idx = c.index[-1]
                if c.loc[idx, 'Adj Close'] > 50.:
                    for col in ['Adj Close', 'Close', 'High', 'Low', 'Open']:
                        data.loc[idx, col] /= 100.
            tu.store_timeseries(data, DATABASE_NAME, STOCKS, stock_id)
        except:
            print(stock_id)
            print(data)


def download_stock_universe(universe):
    u = tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, [universe])
    if u is not None:
        for idx in u.index:
            download_stock_prices(idx)


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
    

def calculate_universe_betas_and_returns(universe, index_ticker, start_date=None, lookback=52, min_obs=26):
    u = tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, [universe])
    if u is not None:
        idx = load_stock_index(index_ticker)
        for stock_id in u.index:
            stock = load_stock_prices(stock_id)
            if stock is not None:
                logger.info('Calculating betas and returns - %s' % stock_id)
                stock = stock['Adj Close']
                stock = stock.ffill(limit=5)
                rtns = pd.concat([idx.resample('W').last(), stock.resample('W').last()], axis=1)
                rtns = rtns.diff() / rtns.shift()
                rtns.columns = ['Index', 'Total']
                if start_date is not None:
                    rtns = rtns[start_date:]
                ans = calculate_beta_and_returns(rtns, lookback, min_obs)
                if ans is not None:
                    ans = ans[['Beta', 'Total', 'Residual', 'Vol']]
                    if ans.Beta.first_valid_index() is not None:
                        tu.store_timeseries(ans[ans.Beta.first_valid_index():], DATABASE_NAME, STOCK_RETURNS, stock_id)


def import_smx_tickers():
    import_tickers('smx', 'SMX Index')


def download_smx_prices():
    download_stock_universe('SMX Index')


def calculate_smx_betas_and_returns(start_date=dt.today() - timedelta(380)):
    calculate_universe_betas_and_returns('SMX Index', 'UKX Index', start_date)


def main():
    download_smx_prices()
    calculate_smx_betas_and_returns()


if __name__ == '__main__':
    main()
