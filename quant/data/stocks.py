'''
Created on 18 Sep 2017

@author: wayne
'''
import os
import numpy as np
import pandas as pd
from quant.lib import data_utils as du, timeseries_utils as tu, portfolio_utils as pu
from quant.lib.main_utils import logger
from quant.data import quandldata
from datetime import datetime as dt


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
    


def import_smx_bloomberg_prices():
    filename = os.path.expanduser('~/TempWork/scripts/smx_bloomberg.xlsx')
    data = pd.read_excel(filename)
    n = len(data.columns)
    i = 0
    while i < n:
        stock_name = data.columns[i+1]
        logger.info('Storing %s' % stock_name)
        tmp = data.iloc[:, i+1]
        tmp.index = data.iloc[:, i]
        tmp = tmp.dropna()
        tmp.name = stock_name
        tmp = tmp.to_frame()
        tu.store_timeseries(tmp, DATABASE_NAME, STOCKS, 'Last Close')
        i += 2


def export_smx_bloomberg_file():
    filename = os.path.expanduser('~/TempWork/scripts/smx_bloomberg.xlsx')
    u = get_smx_universe()
    ans = []
    today = dt.today().strftime('%m/%d/%Y')
    for i, x in enumerate(u.index):
        s = '''=BDH("%s Equity", "PX_LAST", "1/1/2008", "%s")''' % (x, today)
        ans.append(pd.DataFrame([[s, '']], columns=[i+1, x], index=[0]))
    ans = pd.concat(ans, axis=1)
    ff = pd.ExcelWriter(filename)
    ans.to_excel(ff, 'SMX')
    ff.save()


def get_universe(universe):
    return tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, [universe])


def load_stock_prices(start_date=None, end_date=None):
    if start_date is None and end_date is None:
        return tu.get_timeseries(DATABASE_NAME, STOCKS, data_name='Last Close')
    else:
        return tu.get_timeseries(DATABASE_NAME, STOCKS, index_range=(start_date, end_date), data_name='Last Close')


def load_stock_returns(start_date=None, end_date=None, data_name='Returns'):
    if start_date is None and end_date is None:
        return tu.get_timeseries(DATABASE_NAME, STOCK_RETURNS, data_name=data_name)
    else:
        return tu.get_timeseries(DATABASE_NAME, STOCK_RETURNS, index_range=(start_date, end_date), data_name=data_name)


def import_smx_tickers():
    import_tickers('smx', 'SMX Index')


def get_smx_universe():
    return tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, ['SMX Index'])


def calculate_stock_returns():
    u = get_smx_universe()
    p = tu.get_timeseries(DATABASE_NAME, STOCKS, data_name='Last Close')    
    p = p.ffill(limit=5)
    rtns = p.resample('W').last()
    rtns = rtns.diff() / rtns.shift()
    r = rtns.abs()
    vol = r.rolling(52, min_periods=13).median()
    vol[vol < 0.01] = 0.01
    v2 = vol.copy()
    v2[v2 < .03] = .03
    tu.store_timeseries(rtns, DATABASE_NAME, STOCK_RETURNS, 'Returns')
    tu.store_timeseries(vol, DATABASE_NAME, STOCK_RETURNS, 'Volatility')
    tu.store_timeseries(v2, DATABASE_NAME, STOCK_RETURNS, 'PosVol')
