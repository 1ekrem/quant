'''
Created on 18 Sep 2017

@author: wayne
'''
import os
import numpy as np
import pandas as pd
import googlefinance.client as gc
from quant.lib import data_utils as du, timeseries_utils as tu, portfolio_utils as pu
from quant.lib.main_utils import logger
from quant.data import quandldata
from datetime import datetime as dt


DATABASE_NAME = 'quant'
STOCKS_DESCRIPTION = 'stocks_description'
GOOGLE_RETURNS = 'google_returns'


def create_google_table():
    du.create_t2_timeseries_table(DATABASE_NAME, GOOGLE_RETURNS)


# Tickers
def read_tickers_file(filename):
    try: 
        ans = pd.read_excel(os.path.expanduser('~/TempWork/scripts/%s.xlsx' % filename))
        ans.index = ans.ID
        ans = ans.loc[ans.Included == 'Yes']
        return ans.Name
    except:
        return None


def import_tickers(filename, universe):
    data = read_tickers_file(filename)
    if data is not None:
        du.pandas_delete(DATABASE_NAME, STOCKS_DESCRIPTION, du.DESCRIPTION_COLUMN_NAME, du.DESCRIPTION_INDEX_NAME,
                         du.DESCRIPTION_VALUE_NAME, data_name=universe)
        data.name = universe
        tu.store_description(data, DATABASE_NAME, STOCKS_DESCRIPTION)


def import_ftse_smx_tickers():
    import_tickers('FTSESMX', 'SMX')


# Google prices
def load_google_prices(ticker, exchange='LON', period='1Y'):
    param = {'q': ticker, 'i': '86400', 'x': exchange, 'p': period}
    return gc.get_price_data(param)


def import_google_prices(ticker, exchange='LON', period='1Y'):
    logger.info('Loading %s' % ticker)
    data = load_google_prices(ticker, exchange, period)
    if data.empty:
        logger.info('Data not found')
    else:
        data = data.resample('B').last()
        data = data.loc[data.Close > 1e-2]
        v = (data.Close * data.Volume / 1e6).dropna()
        r = np.log(data.Close).diff().dropna()
        v.name = ticker
        r.name = ticker
        tu.store_timeseries(r, DATABASE_NAME, GOOGLE_RETURNS, 'Returns')
        tu.store_timeseries(v, DATABASE_NAME, GOOGLE_RETURNS, 'Volume')


def import_smx_google_prices(period='1M'):
    u = get_ftse_smx_universe()
    for idx in u.index:
        import_google_prices(idx, 'LON', period)


# Load data
def get_universe(universe):
    return tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, [universe])


def get_ftse_smx_universe():
    return tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, ['SMX'])


def load_google_stock_returns(start_date=None, end_date=None, data_name='Returns'):
    if start_date is None and end_date is None:
        return tu.get_timeseries(DATABASE_NAME, GOOGLE_RETURNS, data_name=data_name)
    else:
        return tu.get_timeseries(DATABASE_NAME, GOOGLE_RETURNS, index_range=(start_date, end_date), data_name=data_name)
