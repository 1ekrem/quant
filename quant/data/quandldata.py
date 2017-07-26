'''
Created on 25 Jul 2017

@author: wayne
'''
QUANDL_KEY = "uwzfxL-Kxg5sMVfVuHSW"

import quandl as qd
from quant.lib import data_utils as du, timeseries_utils as tu
from quant.lib.main_utils import logger

qd.ApiConfig.api_key = QUANDL_KEY
DATABASE_NAME = 'quant'
QUANDL_FUTURES = 'quandl_futures'
FUTURES = {'S&P 500': 'CHRIS/CME_SP1',
           'S&P 500 Emini': 'CHRIS/CME_ES1',
           'NASDAQ 200 Emini': 'CHRIS/CME_NQ1',
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
        

# Main function
def download_all_series():
    for series_name, series_id in FUTURES.iteritems():
        download_and_store_series(series_name, series_id, QUANDL_FUTURES)


def main():
    download_all_series()


if __name__ == '__main__':
    main()