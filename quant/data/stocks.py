'''
Created on 18 Sep 2017

@author: wayne
'''
import os
import time
import numpy as np
import pandas as pd
import googlefinance.client as gc
from quant.lib import data_utils as du, timeseries_utils as tu, portfolio_utils as pu
from quant.lib.main_utils import logger
from quant.data import quandldata
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import fix_yahoo_finance as yf


DATABASE_NAME = 'quant'
STOCKS_DESCRIPTION = 'stocks_description'
GLOBAL_ASSETS = 'global_assets'
UK_STOCKS = 'uk_stocks'
SMX_EXCLUDED = ['BGS']
FTSE250_EXCLUDED = ['PIN', 'UKCM']


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


def _save_tickers(data, universe):
    if data is not None:
        du.pandas_delete(DATABASE_NAME, STOCKS_DESCRIPTION, du.DESCRIPTION_COLUMN_NAME, du.DESCRIPTION_INDEX_NAME,
                         du.DESCRIPTION_VALUE_NAME, data_name=universe)
        data.name = universe
        tu.store_description(data, DATABASE_NAME, STOCKS_DESCRIPTION)


def import_tickers(filename, universe):
    data = read_tickers_file(filename)
    _save_tickers(data, universe)


def import_ftse_smx_tickers():
    import_tickers('FTSESMX', 'SMX')


def import_ftse250_tickers():
    import_tickers('FTSE250', 'FTSE250')

# Bloomberg prices
def import_bloomberg_prices(data_type='stock'):
    if data_type == 'stock':
        filename = os.path.expanduser('~/TempWork/scripts/bloomberg.xlsx')
        data_table = UK_STOCKS
        clean_data = True
    elif data_type == 'global':
        filename = os.path.expanduser('~/TempWork/scripts/bloomberg_global.xlsx')
        data_table = GLOBAL_ASSETS
        clean_data = False
    data = pd.read_excel(filename)
    n = len(data.columns)
    i = 0
    while i < n:
        stock_name = data.columns[i+1]
        logger.info('Storing %s' % stock_name)
        tmp = data.iloc[:, i+1]
        tmp.index = data.iloc[:, i]
        tmp = tmp.dropna().to_frame()
        tmp.columns = ['Close']
        save_data(tmp, stock_name, data_table, False, clean_data=clean_data)
        i += 2


def export_smx_bloomberg_file():
    filename = os.path.expanduser('~/TempWork/scripts/bloomberg.xlsx')
    u = get_ftse_smx_universe()
    u2 = get_ftse250_universe()
    u = pd.concat([u, u2.loc[~u2.index.isin(u.index)]], axis=0)
    ans = []
    today = dt.today().strftime('%m/%d/%Y')
    for i, x in enumerate(u.index):
        s = '''=BDH("%s Equity", "PX_LAST", "1/1/2017", "%s")''' % (x, today)
        ans.append(pd.DataFrame([[s, '']], columns=[i+1, x], index=[0]))
    ans = pd.concat(ans, axis=1)
    ff = pd.ExcelWriter(filename)
    ans.to_excel(ff, 'BBG')
    ff.save()


def export_global_bloomberg_file():
    filename = os.path.expanduser('~/TempWork/scripts/bloomberg_global.xlsx')
    ans = []
    today = dt.today().strftime('%m/%d/%Y')
    for i, x in enumerate(['MCX Index', 'GBPUSD BGN Curncy', 'EURUSD BGN Curncy', 'EURGBP BGN Curncy']):
        s = '''=BDH("%s", "PX_LAST", "1/1/2017", "%s")''' % (x, today)
        ans.append(pd.DataFrame([[s, '']], columns=[i+1, x], index=[0]))
    ans = pd.concat(ans, axis=1)
    ff = pd.ExcelWriter(filename)
    ans.to_excel(ff, 'BBG')
    ff.save()


# Google prices
def _divide_hundred(x):
    ans = x.copy()
    ans[ans == 0.] = np.nan
    for i in xrange(len(x)-1):
        if ans.iloc[i+1] / ans.iloc[i] > 70:
            ans.iloc[i+1] /= 100.
        elif ans.iloc[i+1] / ans.iloc[i] > 7:
            ans.iloc[i+1] /= 10.
    return ans
 

def load_yahoo_prices(ticker, start_date=dt(2018,7,1), end_date=dt.today()):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.loc[:, 'Close'] = _divide_hundred(data.loc[:, 'Adj Close'])
        return data
    except Exception as e:
        logger.warn('Failed: %s' % str(e))
        return pd.DataFrame([])


def load_google_prices(ticker, exchange='LON', period='1Y'):
    param = {'q': ticker, 'i': '86400', 'x': exchange, 'p': period}
    try:
        return gc.get_price_data(param)
    except Exception as e:
        logger.warn('Failed: %s' % str(e))
        return pd.DataFrame([])


def save_data(data, ticker, data_table, load_volume=True, clean_data=True):
    data = data.resample('B').last()
    data = data.loc[data.Close > 1e-2]
    r = np.log(data.Close).diff().dropna()
    if clean_data:
        r = tu.remove_outliers(r)
    r.name = ticker
    tu.store_timeseries(r, DATABASE_NAME, data_table, 'Returns')
    if load_volume:
        v = (data.Close * data.Volume / 1e6).dropna()
        v.name = ticker
        tu.store_timeseries(v, DATABASE_NAME, data_table, 'Volume')


def import_yahoo_prices(ticker, save_name, start_date=dt(2018,8,1), end_date=dt.today(), data_table=UK_STOCKS,
                        load_volume=True, clean_data=True):
    logger.info('Loading %s' % ticker)
    data = load_yahoo_prices(ticker, start_date, end_date)
    if data.empty:
        logger.info('Data not found')
    else:
        save_data(data, save_name, data_table, load_volume, clean_data)


def import_google_prices(ticker, exchange='LON', period='1Y', data_table=UK_STOCKS, load_volume=True, clean_data=True):
    logger.info('Loading %s' % ticker)
    data = load_google_prices(ticker, exchange, period)
    if data.empty:
        logger.info('Data not found')
    else:
        save_data(data, ticker, data_table, load_volume, clean_data)


def import_saved_stock_prices():
    filename = os.path.expanduser('~/TempWork/scripts/stocks.xlsx')
    ff = pd.ExcelFile(filename)
    ans = {}
    for stock in ff.sheet_names:
        logger.info('Importing %s' % stock)
        data = ff.parse(stock)
        data.index = data.Date
        data['Close'] = data['Adj Close']
        save_data(data, stock, UK_STOCKS, True, True)
        

def import_ftse250_index_prices(period='1Y'):
    import_google_prices('MCX', 'INDEXFTSE', period, GLOBAL_ASSETS, load_volume=False, clean_data=False)


def import_ftse250_index_prices_from_yahoo(days=30):
    end_date = dt.today()
    start_date = end_date - relativedelta(days=days)
    import_yahoo_prices('^FTMC', 'MCX', start_date, end_date, data_table=GLOBAL_ASSETS, load_volume=False, clean_data=False)


def import_exchange_rates(period='1M'):
    import_google_prices('GBPUSD', 'CURRENCY', period, GLOBAL_ASSETS, load_volume=False, clean_data=False)
    import_google_prices('EURUSD', 'CURRENCY', period, GLOBAL_ASSETS, load_volume=False, clean_data=False)
    import_google_prices('EURGBP', 'CURRENCY', period, GLOBAL_ASSETS, load_volume=False, clean_data=False)
    

def import_exchange_rates_from_yahoo(days=30):
    end_date = dt.today()
    start_date = end_date - relativedelta(days=days)
    import_yahoo_prices('GBPUSD=X', 'GBPUSD', start_date, end_date, data_table=GLOBAL_ASSETS,
                        load_volume=False, clean_data=False)
    import_yahoo_prices('EURUSD=X', 'EURUSD', start_date, end_date, data_table=GLOBAL_ASSETS,
                        load_volume=False, clean_data=False)
    import_yahoo_prices('EURGBP=X', 'EURGBP', start_date, end_date, data_table=GLOBAL_ASSETS,
                        load_volume=False, clean_data=False)
    

def import_uk_google_prices(period='1Y'):
    u = get_ftse_smx_universe()
    u2 = get_ftse250_universe()
    u = pd.concat([u, u2.loc[~u2.index.isin(u.index)]], axis=0)
    for idx in u.index:
        import_google_prices(idx, 'LON', period, UK_STOCKS)


def import_uk_yahoo_prices(years=1, missing=False):
    end_date = dt.today()
    start_date = end_date - relativedelta(years=years)
    u = get_ftse_smx_universe()
    u2 = get_ftse250_universe()
    u = pd.concat([u, u2.loc[~u2.index.isin(u.index)]], axis=0, sort=False)
    if missing:
        r = load_google_returns(dt.today() - relativedelta(days=5), dt.today(), data_table=UK_STOCKS)
        r = r.iloc[-1].loc[u.index]
        u = u.loc[r.isnull()]
    i = 0
    for idx in u.index:
        i += 1
        if i % 30 == 0:
            logger.info('Waiting...')
            time.sleep(60 * 15)
        import_yahoo_prices(idx + '.L', idx, start_date, end_date, data_table=UK_STOCKS,
                            load_volume=True, clean_data=True)


# Load data
def get_universe(universe):
    return tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, [universe])


def get_ftse_smx_universe():
    ans = get_universe('SMX')
    return ans.loc[~ans.index.isin(SMX_EXCLUDED)]


def get_ftse250_universe():
    ans = get_universe('FTSE250')
    return ans.loc[~ans.index.isin(FTSE250_EXCLUDED)]


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
