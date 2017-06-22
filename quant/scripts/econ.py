'''
Created on Aug 7, 2016

@author: Wayne
'''
import numpy as np
import pandas as pd
import fredapi
from yahoo_finance import Share
from datetime import datetime as dt
from matplotlib import pyplot as plt

FREDKEY = 'ff64294203f79127f8d004d2726386ac'
CONFIG = {'NFP TCH Index': ('Friday', 'First'),
          'INJCJC Index': ('Thursday', None),
          'GDP CYOY Index': ('Friday', 'Last', '1,4,7,10'),
          'NAPMPMI Index': ('1'),
          'CPI CHNG Index': ('Thursday', 'Third'),
          'CONCCONF Index': ('Tuesday', 'Last'),
          'CONSSENT Index': ('23'),
          'DGNOCHNG Index': ('Thursday', 'Fourth'),
          'MBAVCHNG Index': ('Wednesday', None),
          'NHSLTOT Index': ('Wednesday', 'Fourth'),
          'NHSPSTOT Index': ('Tuesday', 'Third'),
          'USURTOT Index': ('Friday', 'First'),
          'IP CHNG Index': ('15'),
          'ETSLTOTL Index': ('21'),
          'TMNOCHNG Index': ('6'),
          'PITLCHNG Index': ('M'),
          'PCE CRCH Index': ('M'),
          'USTBTOT Index': ('6'),
          'ADP CHNG Index': ('Wednesday', 'First'),
          'LEI CHNG Index': ('22'),
          'EMPRGBCI Index': ('15'),
          'MWINCHNG Index': ('26'),
          'CNSTTMOM Index': ('1'),
          'OUTFGAF Index': ('Thursday', 'Third')}
WEEKDAYS = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
LOCATION = {'First': 0, 'Second': 1, 'Third': 2, 'Fourth': 3, 'Last': -1}
FREDDATA = [('PAYEMS', 'change'), ('ICSA', 'original'), ('A191RO1Q156NBEA', 'original'), ('CPIAUCSL', 'rate'), ('UMCSENT', 'original'),
            ('DGORDER', 'rate'), ('HSN1F', 'original'), ('HOUST', 'original'), ('UNRATE', 'original'), ('INDPRO', 'rate'), ('AMTMNO', 'rate'),
            ('PI', 'rate'), ('PCE', 'rate'), ('BOPGSTB', 'original'), ('NPPTTL', 'change'), ('USSLIND', 'change'), ('GACDISA066MSFRBNY', 'original'),
            ('WHLSLRMPCIMSA', 'original')]


def get_fred_api():
    return fredapi.Fred(api_key=FREDKEY)


def load_fred_data(series, api, start_date=dt(1990,1,1)):
    data = api.get_series_all_releases(series)
    data = data.loc[data.date >= start_date]
    return data


def get_live_data(data):
    dates = sorted(list(set(data.date)))
    ans = None
    for d in dates:
        tmp = data[data.date == d]
        tmp.index = tmp.realtime_start
        tmp = tmp.value
        if ans is None:
            ans = tmp
        else:
            idx = tmp.index[0]
            ans = pd.concat([ans.loc[[x for x in ans.index if x < idx]], tmp], axis=0)
    return ans


def get_live_change_data(data):
    dates = sorted(list(set(data.date)))
    ans = None
    prev = None
    for d in dates:
        tmp = data[data.date == d]
        tmp.index = tmp.realtime_start
        tmp = tmp.value
        if prev is not None:
            idx = tmp.index[0]
            tmp2 = tmp - prev[:idx].iloc[-1]
            if ans is None:
                ans = tmp2
            else:
                ans = pd.concat([ans.loc[[x for x in ans.index if x < idx]], tmp2], axis=0)
        prev = tmp
    return ans


def get_live_change_rate(data):
    dates = sorted(list(set(data.date)))
    ans = None
    prev = None
    for d in dates:
        tmp = data[data.date == d]
        tmp.index = tmp.realtime_start
        tmp = tmp.value
        if prev is not None:
            idx = tmp.index[0]
            tmp2 = tmp / prev[:idx].iloc[-1] - 1
            if ans is None:
                ans = tmp2
            else:
                ans = pd.concat([ans.loc[[x for x in ans.index if x < idx]], tmp2], axis=0)
        prev = tmp
    return ans


def load_fred_series(api, series, series_type):
    print('Loading %s' % series)
    data = load_fred_data(series, api)
    if series_type == 'original':
        ans = get_live_data(data)
    elif series_type == 'change':
        ans = get_live_change_data(data)
    elif series_type == 'rate':
        ans = get_live_change_rate(data)
    ans.name = series
    return ans


def _load_yahoo_chunk(loader, start_date, end_date):
    print('Loading %d' % start_date.year)
    ToDo = True
    while ToDo:
        try:
            data = loader.get_historical(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            return pd.concat([pd.Series([x['Open'], x['High'], x['Low'], x['Close'], x['Adj_Close'], x['Volume']],
                                        index = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'],
                                        name=dt.strptime(x['Date'], '%Y-%m-%d')).astype(float) for x in data], axis=1).T.sort_index()
        except:
            print('Failed')
            today = dt.today()
            if start_date.year == today.year:
                return None
    

def load_yahoo_prices(ticker, start_date=dt(2000,1,1), end_date=dt.today()):
    x = Share(ticker)
    return _load_yahoo_chunk(x, start_date, end_date)

    
def load_spx_price(start_date=dt(2000,1,1)):
    data = load_yahoo_prices('^GSPC', start_date)
    return data.Adj_Close


def test_fred_economic_data(config=FREDDATA):
    api = get_fred_api()
    for x in config:
        print(load_fred_series(api, *x))


def read_fred_economic_data(config=FREDDATA):
    api = get_fred_api()
    ans = pd.concat([load_fred_series(api, *x) for x in config], axis=1)
    return ans.resample('B').ffill()


def read_bloomberg_data(region='US'):
    data = pd.read_excel('C:\Users\Wayne\Documents\data\data.xlsx', region)
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


def bloomberg_transform(source='release', ECO_RELEASE_DT=None, PX_LAST=None, ACTUAL_RELEASE=None):
    ans = None
    if ACTUAL_RELEASE is not None and ECO_RELEASE_DT is not None and PX_LAST is not None:
        d = ECO_RELEASE_DT.astype('int').astype('str')
        data = pd.concat([d, ACTUAL_RELEASE, PX_LAST.shift(), PX_LAST], axis=1).dropna()
        if source == 'change':
            data.values[:, 1] -= data.values[:, 2]
        elif source == 'revision':
            data.values[:, 1] = data.values[:, 3] - data.shift().values[:, 1]
        data.index = [dt.strptime(x, '%Y%m%d') for x in data.icol(0)]
        ans = data.icol(1)
    return ans


def get_bloomberg_release(data):
    ans = [bloomberg_transform(**v) for v in data.values()]
    return pd.concat([x.resample('B', 'last') for x in ans if x is not None], axis=1).ffill()


def get_bloomberg_change(data):
    ans = [bloomberg_transform('change', **v) for v in data.values()]
    return pd.concat([x.resample('B', 'last') for x in ans if x is not None], axis=1).ffill()


def get_bloomberg_revision(data):
    ans = [bloomberg_transform('revision', **v) for v in data.values()]
    return pd.concat([x.resample('B', 'last') for x in ans if x is not None], axis=1).ffill()
