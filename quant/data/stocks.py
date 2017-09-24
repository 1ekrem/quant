'''
Created on 18 Sep 2017

@author: wayne
'''
import os
import pandas as pd
import fix_yahoo_finance as yf
from datetime import datetime as dt
from quant.lib import data_utils as du, timeseries_utils as tu
from quant.lib.main_utils import logger
from pandas_datareader import data as pdr
from quant.data import quandldata
yf.pdr_override()

DATABASE_NAME = 'quant'
STOCKS = 'stocks'
STOCKS_DESCRIPTION = 'stocks_description'


def create_table():
    du.create_t2_timeseries_table(DATABASE_NAME, STOCKS)
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
    data = load_quandl_stock_prices(stock_id)
    if data is not None:
        tu.store_timeseries(data, DATABASE_NAME, STOCKS, stock_id)


def download_stock_universe(universe):
    u = tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, [universe])
    if u is not None:
        for idx in u.index:
            download_stock_prices(idx)


def load_stock_prices(stock_id, start_date=None, end_date=None):
    return tu.get_timeseries(DATABASE_NAME, STOCKS, index_range=(start_date, end_date), data_name=stock_id)


def import_smx_tickers():
    import_tickers('smx', 'SMX Index')


def download_smx_prices():
    download_stock_universe('SMX Index')


def main():
    download_smx_prices()


if __name__ == '__main__':
    main()
