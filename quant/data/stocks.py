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
GLOBAL_ASSETS = 'global_assets'
UK_STOCKS = 'uk_stocks'


def create_google_table():
    du.create_t2_timeseries_table(DATABASE_NAME, GLOBAL_ASSETS)
    du.create_t2_timeseries_table(DATABASE_NAME, UK_STOCKS)


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


def import_ftse250_tickers():
    import_tickers('FTSE250', 'FTSE250')


# Google prices
def load_google_prices(ticker, exchange='LON', period='1Y'):
    param = {'q': ticker, 'i': '86400', 'x': exchange, 'p': period}
    return gc.get_price_data(param)


def import_google_prices(ticker, exchange='LON', period='1Y', data_table=UK_STOCKS, load_volume=True):
    logger.info('Loading %s' % ticker)
    data = load_google_prices(ticker, exchange, period)
    if data.empty:
        logger.info('Data not found')
    else:
        data = data.resample('B').last()
        data = data.loc[data.Close > 1e-2]
        r = np.log(data.Close).diff().dropna()
        r.name = ticker
        tu.store_timeseries(r, DATABASE_NAME, data_table, 'Returns')
        if load_volume:
            v = (data.Close * data.Volume / 1e6).dropna()
            v.name = ticker
            tu.store_timeseries(v, DATABASE_NAME, data_table, 'Volume')


def import_ftse250_index_prices(period='1M'):
    import_google_prices('MCX', 'INDEXFTSE', period, GLOBAL_ASSETS, load_volume=False)


def import_exchange_rates(period='1M'):
    import_google_prices('GBPUSD', 'CURRENCY', period, GLOBAL_ASSETS, load_volume=False)
    import_google_prices('EURUSD', 'CURRENCY', period, GLOBAL_ASSETS, load_volume=False)
    import_google_prices('EURGBP', 'CURRENCY', period, GLOBAL_ASSETS, load_volume=False)


def import_smx_google_prices(period='1M'):
    u = get_ftse_smx_universe()
    for idx in u.index:
        import_google_prices(idx, 'LON', period, UK_STOCKS)


def import_ftse250_google_prices(period='1M'):
    u = get_ftse250_universe()
    for idx in u.index:
        import_google_prices(idx, 'LON', period, UK_STOCKS)


# Load data
def get_universe(universe):
    return tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, [universe])


def get_ftse_smx_universe():
    return tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, ['SMX'])


def get_ftse250_universe():
    return tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, ['FTSE250'])


def load_google_returns(start_date=None, end_date=None, data_name='Returns', data_table=GLOBAL_ASSETS):
    if start_date is None and end_date is None:
        return tu.get_timeseries(DATABASE_NAME, data_table, data_name=data_name)
    else:
        return tu.get_timeseries(DATABASE_NAME, data_table, index_range=(start_date, end_date), data_name=data_name)


def create_universe():
    create_google_table()
    import_ftse_smx_tickers()
    import_ftse250_tickers()
    import_ftse250_index_prices('20Y')
    import_exchange_rates('20Y')
    import_smx_google_prices('20Y')
    import_ftse250_google_prices('20Y')
