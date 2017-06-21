'''
Created on Jun 1, 2017

@author: Wayne
'''
import numpy as np
import pandas as pd
import MySQLdb as mdb
import logging
from datetime import datetime as dt
from matplotlib.rcsetup import _seq_err_msg
from pandas_datareader.io.sdmx import _SERIES

# Database info
HOST = 'localhost'
USER = 'wayne'
PASSWORD = ''

# SQL COMMANDS
CREATE_TABLE_IF_NOT_EXISTS = "CREATE TABLE IF NOT EXISTS %s (%s);"
TIMESERIES_TABLE_FORMAT = "time_index DATETIME primary key, column_name VARCHAR(20), value FLOAT"
BULK_TABLE_INSERT = "INSERT INTO %s %s VALUES %s;"
BULK_TABLE_DELETE = "DELETE FROM %s WHERE %s in (%s);"


def _series_to_insert_sql(s):
    name = s.name
    data = s.dropna().astype('str')
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.strftime('%Y-%m-%d %H:%M:%S')
    return ', '.join(["('%s', '%s', '%s')" % (k, name, v) for k, v in data.to_dict().iteritems()])


def _series_to_delete_sql(s):
    name = s.name
    data = s.dropna().astype('str')
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.strftime('%Y-%m-%d %H:%M:%S')
    return ', '.join(["('%s', '%s')" % (k, name) for k in data.index])


def get_insert_sql_from_pandas(data):
    if isinstance(data, pd.Series):
        return _series_to_insert_sql(data)
    elif isinstance(data, pd.DataFrame):
        ans = [_series_to_insert_sql(data.iloc[:, i]) for i in xrange(np.size(data, 1))]
        return ', '.join([s for s in ans if len(s)>0])
    else:
        return ''


def get_delete_sql_from_pandas(data):
    if isinstance(data, pd.Series):
        return _series_to_delete_sql(data)
    elif isinstance(data, pd.DataFrame):
        ans = [_series_to_delete_sql(data.iloc[:, i]) for i in xrange(np.size(data, 1))]
        return ', '.join([s for s in ans if len(s)>0])
    else:
        return ''


def get_pandas_bulk_insert_script(data, table_name, column_name, index_name, value_name):
    insert_value_script = get_insert_sql_from_pandas(data)
    insert_format_script = '(%s, %s, %s)' % (index_name, column_name, value_name)
    if len(insert_value_script) > 0:
        insert_script = BULK_TABLE_INSERT % (table_name, insert_format_script, insert_value_script)
    else:
        insert_script = ''
    return insert_script


def get_pandas_bulk_delete_script(data, table_name, column_name, index_name, value_name):
    delete_value_script = get_delete_sql_from_pandas(data)
    delete_format_script = '(%s, %s)' % (index_name, column_name)
    if len(delete_value_script) > 0:
        delete_script = BULK_TABLE_DELETE % (table_name, delete_format_script, delete_value_script)
    else:
        delete_script = ''
    return delete_script


# Database i/o
def get_database_connection(database_name='mysql'):
    try:
        return mdb.connect(host=HOST, user=USER, passwd=PASSWORD, db=database_name)
    except Exception as e:
        logging.warning('Failed to establish databse connection: ' + str(e))
        return None


def execute_sql_script(database_name, scripts):
    db = get_database_connection(database_name)
    if db is not None:
        try:
            cur = db.cursor()
            cur.execute(scripts)
            db.commit()
            db.close()
            return None
        except Exception as e:
            return e


def create_timeseries_table(database_name, table_name):
    script = CREATE_TABLE_IF_NOT_EXISTS % (table_name, TIMESERIES_TABLE_FORMAT)
    e = execute_sql_script(database_name, script)
    if e is not None:
        logging.warning('Failed to create table: ' + str(e))



