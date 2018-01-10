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
        for c in data.columns:
            tmp = data[c].to_frame()
            tmp.columns = [stock_id]
            tu.store_timeseries(tmp, DATABASE_NAME, STOCKS, c)


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


def get_smx_universe():
    u = tu.get_description(DATABASE_NAME, STOCKS_DESCRIPTION, ['SMX Index'])
    idx = []
    for y, x in u.iloc[:, 0].to_dict().iteritems():
        if 'TRUST' not in x and 'FUND' not in x and 'BLACKROCK' not in x and 'FIDELITY' not in x \
        and 'ABERDEEN' not in x and 'ALPHA' not in x and 'BARING' not in x and 'BH ' not in x \
        and 'BAILLIE' not in x and 'F&C' not in x and 'INV TR' not in x and 'HENDERSON' not in x \
        and 'JPMORGAN' not in x and 'MONTANARO' not in x and 'POLAR' not in x and 'SCHRODER' not in x \
        and 'STANDARD LIFE' not in x and 'INCOME' not in x and 'IMPAX' not in x and 'VINACAPITAL' not in x \
        and 'HIGHBRIDGE' not in x and 'EDISTON' not in x and 'PRIVATE EQUITY' not in x and 'ECOFIN' not in x \
        and 'MARTIN' not in x and 'INVESCO' not in x and 'TWENTYFOUR' not in x and 'REAL ESTATE' not in x \
        and 'DUNEDIN' not in x and 'JUPITER' not in x and 'BBGI' not in x and 'REIT' not in x \
        and 'WITAN' not in x and 'DEBENTURE' not in x:
            idx.append(y)
    return u.loc[idx]
    
    
def download_smx_prices():
    u = get_smx_universe()
    for idx in u.index:
        download_stock_prices(idx)


def calculate_stock_returns():
    u = get_smx_universe()
    p = tu.get_timeseries(DATABASE_NAME, STOCKS, data_name='Last Close')    
    p = p.ffill(limit=5)
    rtns = p.resample('W').last()
    rtns = rtns.diff() / rtns.shift()
    r = rtns.abs()
    vol = r[r>0].rolling(26, min_periods=8).median() * np.sqrt(52.)
    tu.store_timeseries(rtns, DATABASE_NAME, STOCK_RETURNS, 'Returns')
    tu.store_timeseries(vol, DATABASE_NAME, STOCK_RETURNS, 'Volatility')


def main():
    download_smx_prices()
    calculate_stock_returns()


if __name__ == '__main__':
    main()
