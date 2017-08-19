'''
Created on 25 Jul 2017

@author: wayne
'''
QUANDL_KEY = "uwzfxL-Kxg5sMVfVuHSW"

import pandas as pd
import quandl as qd
from quant.lib import data_utils as du, timeseries_utils as tu
from quant.lib.main_utils import logger

qd.ApiConfig.api_key = QUANDL_KEY
DATABASE_NAME = 'quant'
QUANDL_FUTURES = 'quandl_futures'
FUTURES = {'S&P 500': 'CHRIS/CME_SP1',
           'S&P 500 Emini': 'CHRIS/CME_ES1',
           'NASDAQ 200 Emini': 'CHRIS/CME_NQ1',
           'Eurostoxx 50': 'CHRIS/EUREX_FESX1',
           'US Treasury': 'CHRIS/CME_US1',
           'Crude Oil': 'CHRIS/CME_CL1',
           'Eurodollar': 'CHRIS/CME_ED1',
           'Copper': 'CHRIS/CME_HG1',
           'EUR/USD': 'CHRIS/CME_EC1',
           'GBP/USD': 'CHRIS/CME_BP1',
           'AUD/USD': 'CHRIS/CME_AD1',
           'NZD/USD': 'CHRIS/CME_NE1',
           'US Dollar': 'CHRIS/ICE_DX1',
           'Gold': 'CHRIS/CME_GC1',
           }


# SQL utils
def create_table():
    du.create_t2_timeseries_table(DATABASE_NAME, QUANDL_FUTURES)


# Loading data
def load_series(series_id):
    try:
        return qd.get(series_id)
    except Exception as e:
        logger.warn('Failed to load quandl data %s: %s' % (series_id, str(e)))
        return None
    

def download_and_store_series(series_name, series_id, table_name=QUANDL_FUTURES):
    logger.info('Downloading quandl series %s %s' % (series_name, series_id))
    series = load_series(series_id)
    if series is not None:
        logger.info('Storing quandl series - %s' % series_name)
        tu.store_timeseries(series, DATABASE_NAME, table_name, series_name)
        

def get_quandl_series(series_name, start_date=None, end_date=None):
    ans = tu.get_timeseries(DATABASE_NAME, QUANDL_FUTURES, index_range=(start_date, end_date),
                            column_list=['Settle'], data_name=series_name)
    if ans is not None:
        ans = ans.iloc[:, 0]
        ans.name = series_name
        return ans
    else:
        return None


def quandl_price_loader(series, start_date=None, end_date=None):
    ans = []
    for series_name in series:
        data = get_quandl_series(series_name, start_date, end_date)
        if data is not None:
            ans.append(data)
    return pd.concat(ans, axis=1) if len(ans)>0 else None
    
# Main function
def download_all_series():
    for series_name, series_id in FUTURES.iteritems():
        download_and_store_series(series_name, series_id, QUANDL_FUTURES)


def main():
    download_all_series()


if __name__ == '__main__':
    main()