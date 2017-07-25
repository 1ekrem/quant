'''
Created on 25 Jul 2017

@author: wayne
'''
QUANDL_KEY = "uwzfxL-Kxg5sMVfVuHSW"

import quandl as qd
from quant.lib import data_utils as du
from quant.lib.main_utils import logger

qd.ApiConfig.api_key = QUANDL_KEY
DATABASE_NAME = 'quant'
QUANDL_FUTURES = 'quandl_futures'
TABLE_FORMAT = "time_index DATETIME, series_name VARCHAR(50), series_type VARCHAR(50), value DOUBLE"

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
    du.create_table(DATABASE_NAME, QUANDL_FUTURES, TABLE_FORMAT)


def _insert_sql(s, series_name):
    data = s.copy()
    data.index = data.index.strftime(du.DATETIME_FORMAT)
    ans = []
    for c in data.columns:
        sb = data[c].dropna()
        for idx in sb.index:
            ans.append("('%s', '%s', '%s', '%f')" % (idx, series_name, c, sb.loc[idx]))
    return ', '.join(ans)


def _delete_sql(s, series_name):
    data = s.copy()
    data.index = data.index.strftime(du.DATETIME_FORMAT)
    ans = []
    for c in data.columns:
        for idx in data.index:
            ans.append("('%s', '%s', '%s')" % (idx, series_name, c))
    return ', '.join(ans)


def get_bulk_insert_script(data, table_name, series_name):
    insert_value_script = _insert_sql(data, series_name)
    insert_format_script = "(time_index, series_name, series_type, value)"
    if len(insert_value_script) > 0:
        insert_script = du.BULK_TABLE_INSERT % (table_name, insert_format_script, insert_value_script)
    else:
        insert_script = ''
    return insert_script


def get_bulk_delete_script(data, table_name, series_name):
    delete_value_script = _delete_sql(data, series_name)
    delete_format_script = "(time_index, series_name, series_type)"
    if len(delete_value_script) > 0:
        delete_script = du.BULK_TABLE_DELETE % (table_name, delete_format_script, delete_value_script)
    else:
        delete_script = ''
    return delete_script


def get_read_script(table_name, series_name, index_range=None, column_list=None):
    read_format_script = 'series_type, time_index, value'
    condition_script = du.get_pandas_select_condition_script('time_index', 'series_type', index_range, column_list)
    if len(condition_script) > 0:
        condition_script += " AND series_name = '%s'" % series_name
    else:
        condition_script += "WHERE series_name = '%s'" % series_name
    return du.READ_TABLE % (read_format_script, table_name, condition_script)


def bulk_insert(data, table_name, series_name):
    delete_script = get_bulk_delete_script(data, table_name, series_name)
    e = du.execute_sql_input_script(DATABASE_NAME, delete_script)
    if e is not None:
        logger.warning('Failed to clear data from table: ' + str(e))
    else:
        insert_script = get_bulk_insert_script(data, table_name, series_name)
        e = du.execute_sql_input_script(DATABASE_NAME, insert_script)
        if e is not None:
            logger.warning('Failed to insert data: ' + str(e))


# Loading data
def load_series(series_id):
    try:
        return qd.get(series_id)
    except Exception as e:
        logger.warn('Failed to load quandl data %s: %s' % (series_id, str(e)))
        return None


def store_series(data, series_name, table_name=QUANDL_FUTURES):
    logger.info('Storing series - %s' % series_name)
    bulk_insert(data, table_name, series_name)


def get_series(series_name, start_date=None, end_date=None, column_list=None, table_name=QUANDL_FUTURES):
    read_script = get_read_script(table_name, series_name, index_range=(start_date, end_date), column_list=column_list)
    success, data = du.execute_sql_output_script(DATABASE_NAME, read_script)
    if success:
        return du.get_pandas_output(data, 'series_type', 'time_index', 'value') if len(data)>0 else None
    else:
        logger.warning('Failed to read data: ' + str(data))


def download_and_store_series(series_name, series_id, table_name=QUANDL_FUTURES):
    logger.info('Downloading quandl series %s %s' % (series_name, series_id))
    series = load_series(series_id)
    if series is not None:
        store_series(series, series_name, table_name)


# Main function
def download_all_series():
    for series_name, series_id in FUTURES.iteritems():
        download_and_store_series(series_name, series_id, QUANDL_FUTURES)


def main():
    download_all_series()


if __name__ == '__main__':
    main()