'''
Created on 18 Sep 2017

@author: wayne
'''
from quant.lib.main_utils import *
import googlefinance.client as gc
from quant.lib import data_utils as du, timeseries_utils as tu, portfolio_utils as pu, web_utils as wu
from quant.data import quandldata
import fix_yahoo_finance as yf


DATABASE_NAME = 'quant'
STOCKS_DESCRIPTION = 'stocks_description'
UK_STOCKS = 'uk_stocks'
UK_FINANCIALS = 'uk_financials'
UK_ESTIMATES = 'uk_estimates'
SMX_EXCLUDED = ['BGS']
FTSE250_EXCLUDED = ['PIN', 'UKCM']
FTSE100_EXCLUDED = []
AIM_EXCLUDED = []


def create_google_table():
    du.create_table(DATABASE_NAME, STOCKS_DESCRIPTION, du.DESCRIPTION_TABLE_FORMAT)
    du.create_t2_timeseries_table(DATABASE_NAME, UK_STOCKS)
    du.create_t2_timeseries_table(DATABASE_NAME, UK_FINANCIALS)
    du.create_t2_timeseries_table(DATABASE_NAME, UK_ESTIMATES)


# Tickers
def _save_tickers(data, universe):
    if data is not None:
        data.name = universe
        tu.store_description(data, DATABASE_NAME, STOCKS_DESCRIPTION)


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
 

def _remove_zero(x):
    return x.loc[~(x == 0.).any(axis=1)]


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


def load_av_prices(ticker, output_size='compact'):
    ans = pd.DataFrame([])
    try:
        data = wu.load_alpha_vantage(ticker, output_size=output_size)
        if data is not None:
            ans = pd.DataFrame(data).T
            ans.index = pd.DatetimeIndex(ans.index)
            ans.loc[:, 'Close'] = ans.loc[:, '5. adjusted close'].astype(float)
            ans.loc[:, 'Volume'] = ans.loc[:, '6. volume'].astype(float)
            ans = _remove_zero(ans[['Close', 'Volume']])
    except Exception as e:
        logger.warn('Failed: %s' % str(e))
    return ans
        

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


def import_uk_yahoo_prices(years=1, missing=False):
    end_date = dt.today()
    start_date = end_date - relativedelta(years=years)
    u = load_uk_universe()
    if missing:
        r = load_google_returns(dt.today() - relativedelta(days=5), dt.today(), data_table=UK_STOCKS)
        r = r.iloc[-1].reindex(u.index)
        u = u.loc[r.isnull()]
    i = 0
    for idx in u.index:
        i += 1
        if i % 25 == 0:
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


def get_ftse100_universe():
    ans = get_universe('FTSE100')
    return ans.loc[~ans.index.isin(FTSE100_EXCLUDED)]


def get_ftse_aim_universe():
    ans = get_universe('AIM')
    return ans.loc[~ans.index.isin(AIM_EXCLUDED)]


def load_universe(universe):
    if universe == 'SMX':
        u = get_ftse_smx_universe()
    elif universe == 'FTSE250':
        u = get_ftse250_universe()
    elif universe == 'AIM':
        u = get_ftse_aim_universe()
    elif universe == 'FTSE100':
        u = get_ftse100_universe()
    return u


def load_uk_universe():
    u = get_ftse_smx_universe()
    u2 = get_ftse250_universe()
    u3 = get_ftse_aim_universe()
    u4 = get_ftse100_universe()
    u = pd.concat([u, u2.loc[~u2.index.isin(u.index)]], axis=0, sort=False)
    u = pd.concat([u, u3.loc[~u3.index.isin(u.index)]], axis=0, sort=False)
    u = pd.concat([u, u4.loc[~u3.index.isin(u.index)]], axis=0, sort=False)
    return u


def load_google_returns(start_date=None, end_date=None, tickers=None, data_name='Returns', data_table=UK_STOCKS):
    if start_date is None and end_date is None:
        return tu.get_timeseries(DATABASE_NAME, data_table, column_list=tickers, data_name=data_name)
    else:
        return tu.get_timeseries(DATABASE_NAME, data_table, column_list=tickers,
                                 index_range=(start_date, end_date), data_name=data_name)


def load_financial_data(data_name='revenue', tickers=None, data_table=UK_FINANCIALS):
    return tu.get_timeseries(DATABASE_NAME, data_table, column_list=tickers, data_name=data_name)


