'''
Created on 6 Jul 2017

@author: wayne
'''
import sys
import fredapi
import numpy as np
import pandas as pd
from quant.lib import data_utils as du
from quant.lib.main_utils import logger

DATABASE_NAME = 'quant'
INFO_TABLE_NAME = 'fred_econ_info'
SERIES_TABLE_NAME = 'fred_data'
RELEASE_TABLE_NAME = 'fred_econ_data'
TABLE_FORMAT = "time_index DATETIME, realtime_start DATETIME, series_name VARCHAR(50), value FLOAT"
INFO_TABLE_FORMAT = "series_name VARCHAR(50), description VARCHAR(50), value VARCHAR(1000)"
FREDKEY = 'ff64294203f79127f8d004d2726386ac'
_api = fredapi.Fred(api_key=FREDKEY)

# data config
US_ECON = [# Economic indicator
           'CPIAUCSL', 'T10YIE', 'T5YIFR', 'PAYEMS', 'USSLIND',
           # National income
           'GDP', 'GDPC1',
           # Interest rates
           'WGS10YR', 'WGS5YR', 'WGS2YR', 'WGS1YR', 'MORTGAGE30US', 'FF', 'DTB3', 'FEDFUNDS', 'T10Y2Y',
           'USD3MTD156N', 'USD1MTD156N', 'BAMLH0A0HYM2EY', 'TEDRATE',
           # Current population survey
           'UNRATE',
           # Housing
           'HOUST', 'HOUST1F', 'CSUSHPINSA',
           # Industrial production
           'INDPRO',
           # Transportation
           'RAILFRTINTERMODAL',
           # Corporate bond yield
           'WAAA', 'WBAA',
           # Risk indicator
           'DRSFRMACBS', 'BAMLH0A0HYM2',
           ]
EU_ECON = [# Interest rates
           'IRLTLT01DEM156N',
           ]
CHINA_ECON = [# Economic indicator
              'CHNCPIALLMINMEI'
              ]
US_SERIES = [# Stock
             'SP500', 'NASDAQCOM',
             # FX
             'DEXUSEU', 'DEXUSUK', 'DEXUSNZ', 'DEXUSAL', 'DEXJPUS',
             # Commodities
             'DCOILWTICO']

# utils
def create_tables():
    du.create_timeseries_table(DATABASE_NAME, SERIES_TABLE_NAME)
    du.create_table(DATABASE_NAME, RELEASE_TABLE_NAME, TABLE_FORMAT)
    du.create_table(DATABASE_NAME, INFO_TABLE_NAME, INFO_TABLE_FORMAT)


def load_series_info(series_name):
    logger.info('Loading series info - %s' % series_name)
    try:
        return _api.get_series_info(series_name)
    except Exception as e:
        logger.warn('Failed to get series info: %s' % str(e))
        return None


def load_series(series_name):
    logger.info('Loading series - %s' % series_name)
    try:
        return _api.get_series(series_name)
    except Exception as e:
        logger.warn('Failed to get series: %s' % str(e))
        return None


def load_series_all_releases(series_name):
    logger.info('Loadind series all release - %s' % series_name)
    try:
        return _api.get_series_all_releases(series_name)
    except Exception as e:
        logger.warn('Failed to get series all releases: %s' % str(e))
        return None


def _encode_string(s):
    return s.encode('utf-8').replace('"', "'")[:1000]


def store_series_info(data, series_name):
    logger.info('Storing series info - %s' % series_name)
    series = data.copy().apply(_encode_string)
    series.name = series_name
    du.pandas_bulk_insert(series, DATABASE_NAME, INFO_TABLE_NAME, 'series_name', 'description', 'value')


def store_series(data, series_name):
    logger.info('Storing series - %s' % series_name)
    series = data.copy()
    series.name = series_name
    du.pandas_bulk_insert(series, DATABASE_NAME, SERIES_TABLE_NAME, du.TIMESERIES_COLUMN_NAME, du.TIMESERIES_INDEX_NAME, du.TIMESERIES_VALUE_NAME)


def _release_to_insert_sql(s, series_name):
    data = s.dropna().astype('str')
    data['series_name'] = series_name
    data = data[['date', 'realtime_start', 'series_name', 'value']]
    return ', '.join([str(tuple(v)) for v in data.values])


def _release_to_delete_sql(s, series_name):
    data = s.dropna().astype('str')
    data['series_name'] = series_name
    data = data[['date', 'realtime_start', 'series_name']]
    return ', '.join([str(tuple(v)) for v in data.values])


def get_release_bulk_insert_script(data, table_name, series_name):
    insert_value_script = _release_to_insert_sql(data, series_name)
    insert_format_script = "(time_index, realtime_start, series_name, value)"
    if len(insert_value_script) > 0:
        insert_script = du.BULK_TABLE_INSERT % (table_name, insert_format_script, insert_value_script)
    else:
        insert_script = ''
    return insert_script


def get_release_bulk_delete_script(data, table_name, series_name):
    delete_value_script = _release_to_delete_sql(data, series_name)
    delete_format_script = "('time_index', 'realtime_start', 'series_name')"
    if len(delete_value_script) > 0:
        delete_script = du.BULK_TABLE_DELETE % (table_name, delete_format_script, delete_value_script)
    else:
        delete_script = ''
    return delete_script


def release_bulk_insert(data, table_name, series_name):
    delete_script = get_release_bulk_delete_script(data, table_name, series_name)
    e = du.execute_sql_input_script(DATABASE_NAME, delete_script)
    if e is not None:
        logger.warning('Failed to clear data from table: ' + str(e))
    else:
        insert_script = get_release_bulk_insert_script(data, table_name, series_name)
        e = du.execute_sql_input_script(DATABASE_NAME, insert_script)
        if e is not None:
            logger.warning('Failed to insert data: ' + str(e))


def store_series_all_releases(data, series_name):
    logger.info('Storing series all release - %s' % series_name)
    release_bulk_insert(data, RELEASE_TABLE_NAME, series_name)


def download_and_store_series(series_name):
    series_info = load_series_info(series_name)
    if series_info is not None:
        store_series_info(series_info, series_name)
    series = load_series(series_name)
    if series is not None:
        store_series(series, series_name)


def download_and_store_series_all_releases(series_name):
    series_info = load_series_info(series_name)
    if series_info is not None:
        store_series_info(series_info, series_name)
    series = load_series_all_releases(series_name)
    if series is not None:
        store_series_all_releases(series, series_name)


def get_series_info(series_name):
    return du.pandas_read(DATABASE_NAME, INFO_TABLE_NAME, 'series_name', 'description', 'value', column_list=[series_name])


def get_series(series_name, start_date=None, end_date=None):
    return du.pandas_read(DATABASE_NAME, SERIES_TABLE_NAME, du.TIMESERIES_COLUMN_NAME, du.TIMESERIES_INDEX_NAME,
                          du.TIMESERIES_VALUE_NAME, index_range=(start_date, end_date), column_list=[series_name])


def get_release_read_script(table_name, index_range=None, column_list=None):
    read_format_script = 'time_index, realtime_start, series_name, value'
    condition_script = du.get_pandas_select_condition_script('time_index', 'series_name', index_range, column_list)
    return du.READ_TABLE % (read_format_script, table_name, condition_script)


def get_series_all_release(series_name, start_date=None, end_date=None):
    read_script = get_release_read_script(RELEASE_TABLE_NAME, index_range=(start_date, end_date), column_list=[series_name])
    success, data = du.execute_sql_output_script(DATABASE_NAME, read_script)
    if success:
        return pd.DataFrame(np.array(data), columns=['time_index', 'realtime_start', 'series_name', 'value']) if len(data)>0 else None
    else:
        logger.warning('Failed to read data: ' + str(data))


def download_all_releases():
    for series_name in US_ECON + CHINA_ECON + EU_ECON:
        download_and_store_series_all_releases(series_name)


def download_all_series():
    for series_name in US_SERIES:
        download_and_store_series(series_name)


def main():
    if len(sys.argv) == 1:
        download_all_releases()
        download_all_series()
    elif len(sys.argv) == 2:
        if sys.argv[1] == 'series':
            download_all_series()
        elif sys.argv[1] == 'release':
            download_all_releases()
    else:
        s1 = sys.argv[1]
        s2 = sys.argv[2]
        if s1 == 'series':
            download_and_store_series(s2)
        elif s1 == 'release':
            download_and_store_series_all_releases(s2)


if __name__ == '__main__':
    main()
