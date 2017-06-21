'''
Created on Jun 1, 2017

@author: Wayne
'''
import numpy as np
import pandas as pd
from datetime import datetime as dt
from yahoo_finance import Share


def load_share_by_year_from_yahoo(ticker, loader, start_date, end_date, return_partial_data=False):
    year1 = start_date.year
    year2 = end_date.year
    if year1 < year2:
        years_to_load = np.arange(year1, year2+1)
    else:
        years_to_load = np.array([year1])
    ans = []
    success = True
    for year in years_to_load:
        load_count = 1
        ToLoad = True
        while ToLoad and load_count < 4:
            try:
                data = loader.get_historical('%d-01-01' % year, '%d-12-31' % year)
                print(data)
                data = pd.concat([pd.Series([x['Open'], x['High'], x['Low'], x['Close'], x['Adj_Close'], x['Volume']],
                                            index = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'],
                                            name=dt.strptime(x['Date'], '%Y-%m-%d')).astype(float) for x in data], axis=1).T.sort_index()
                print('Successful: Loading %s %d for %d time' % (ticker, year, load_count))
            except:
                print('Failed: Loading %s %d for %d time' % (ticker, year, load_count))
                load_count += 1
                data = None
        if data is not None:
            ans.append(data)
        else:
            success = False
            if not return_partial_data:
                break
    if success or return_partial_data:
        return pd.concat(ans, axis=0).sort_index()
    else:
        return None


def load_share_from_yahoo(ticker, start_date, end_date, return_partial_data=False):
    loader = Share(ticker)
    return load_share_by_year_from_yahoo(ticker, loader, start_date, end_date, return_partial_data)

    