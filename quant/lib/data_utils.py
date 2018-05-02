'''
Created on Jun 1, 2017

@author: Wayne
'''
import numpy as np
import pandas as pd
import MySQLdb as mdb
from quant.lib.main_utils import logger
from datetime import datetime as dt

# Database info
HOST = 'localhost'
USER = 'root'
PASSWORD = ''

# SQL COMMANDS
TIMESERIES_INDEX_NAME = 'time_index'
TIMESERIES_COLUMN_NAME = 'column_name'
TIMESERIES_DATA_NAME = 'data_name'
TIMESERIES_VALUE_NAME = 'value'
DESCRIPTION_INDEX_NAME = 'column_name'
DESCRIPTION_COLUMN_NAME = 'data_name'
DESCRIPTION_VALUE_NAME = 'value'
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CREATE_TABLE_IF_NOT_EXISTS = "CREATE TABLE IF NOT EXISTS %s (%s);"
TIMESERIES_TABLE_FORMAT = "time_index DATETIME, column_name VARCHAR(50), value DOUBLE"
T2_TIMESERIES_TABLE_FORMAT = "time_index DATETIME, data_name VARCHAR(50), column_name VARCHAR(50), value DOUBLE"
DESCRIPTION_TABLE_FORMAT = "data_name VARCHAR(50), column_name VARCHAR(50), value VARCHAR(300)"
BULK_TABLE_INSERT = "INSERT INTO %s %s VALUES %s;"
BULK_TABLE_DELETE = "DELETE FROM %s WHERE %s in (%s);"
DELETE_DATA = "DELETE FROM %s %s;"
READ_TABLE = "SELECT %s FROM %s %s;"
READ_COLUMN_VALUES = "SELECT DISTINCT(%s) FROM %s;"


def _series_to_insert_sql(s, data_name=None):
    name = s.name
    data = s.dropna().astype('str')
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.strftime(DATETIME_FORMAT)
    if data_name is None:
        return ', '.join(['''("%s", "%s", "%s")''' % (k, name, v) for k, v in data.to_dict().iteritems()])
    else:
        return ', '.join(['''("%s", "%s", "%s", "%s")''' % (k, data_name, name, v) for k, v in data.to_dict().iteritems()])


def _series_to_delete_sql(s, data_name=None):
    name = s.name
    data = s.dropna().astype('str')
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.strftime(DATETIME_FORMAT)
    if data_name is None:
        return ', '.join(['''("%s", "%s")''' % (k, name) for k in data.index])
    else:
        return ', '.join(['''("%s", "%s", "%s")''' % (k, data_name, name) for k in data.index])


def get_insert_sql_from_pandas(data, data_name=None):
    if isinstance(data, pd.Series):
        return _series_to_insert_sql(data, data_name)
    elif isinstance(data, pd.DataFrame):
        ans = [_series_to_insert_sql(data.iloc[:, i], data_name) for i in xrange(np.size(data, 1))]
        return ', '.join([s for s in ans if len(s)>0])
    else:
        return ''


def get_delete_sql_from_pandas(data, data_name=None):
    if isinstance(data, pd.Series):
        return _series_to_delete_sql(data, data_name)
    elif isinstance(data, pd.DataFrame):
        ans = [_series_to_delete_sql(data.iloc[:, i], data_name) for i in xrange(np.size(data, 1))]
        return ', '.join([s for s in ans if len(s)>0])
    else:
        return ''


def get_pandas_bulk_insert_script(data, table_name, column_name, index_name, value_name, data_name=None, data_column_name=None):
    '''
    Built to handle T1 and T2 timeseries format
    '''
    insert_value_script = get_insert_sql_from_pandas(data, data_name)
    if data_name is None:
        insert_format_script = '(%s, %s, %s)' % (index_name, column_name, value_name)
    else:
        insert_format_script = '(%s, %s, %s, %s)' % (index_name, data_column_name, column_name, value_name)
    if len(insert_value_script) > 0:
        insert_script = BULK_TABLE_INSERT % (table_name, insert_format_script, insert_value_script)
    else:
        insert_script = ''
    return insert_script


def get_pandas_bulk_delete_script(data, table_name, column_name, index_name, data_name=None, data_column_name=None):
    '''
    Built to handle T1 and T2 timeseries format
    '''
    delete_value_script = get_delete_sql_from_pandas(data, data_name)
    if data_name is None:
        delete_format_script = '(%s, %s)' % (index_name, column_name)
    else:
        delete_format_script = '(%s, %s, %s)' % (index_name, data_column_name, column_name)
    if len(delete_value_script) > 0:
        delete_script = BULK_TABLE_DELETE % (table_name, delete_format_script, delete_value_script)
    else:
        delete_script = ''
    return delete_script


def get_pandas_select_condition_script(index_name, column_name, index_range, column_list, data_name=None):
    conditions = []
    if data_name is not None:
        conditions.append("data_name = '%s'" % data_name)
    if isinstance(index_range, tuple) and len(index_range) == 2:
        lower, higher = index_range
        if lower is not None:
            conditions.append(index_name + " >= '%s'" % lower.strftime(DATETIME_FORMAT) if isinstance(lower, dt) else str(lower))
        if higher is not None:
            conditions.append(index_name + " <= '%s'" % higher.strftime(DATETIME_FORMAT) if isinstance(higher, dt) else str(higher))
    if column_list is not None:
        conditions.append(column_name + ' IN (' + ', '.join(["'" + str(item) + "'" for item in column_list]) + ')')
    if len(conditions) > 0:
        return 'WHERE ' + ' AND '.join(conditions)
    else:
        return ''


def get_pandas_read_script(table_name, column_name, index_name, value_name, index_range=None, column_list=None, data_name=None):
    read_format_script = '%s, %s, %s' % (column_name, index_name, value_name)
    condition_script = get_pandas_select_condition_script(index_name, column_name, index_range, column_list, data_name)
    return READ_TABLE % (read_format_script, table_name, condition_script)


def get_pandas_delete_script(table_name, column_name, index_name, value_name, index_range=None, column_list=None, data_name=None):
    condition_script = get_pandas_select_condition_script(index_name, column_name, index_range, column_list, data_name)
    return DELETE_DATA % (table_name, condition_script)


def get_table_column_value_scripts(table_name, column_name=TIMESERIES_COLUMN_NAME):
    return READ_COLUMN_VALUES % (column_name, table_name)


# Database i/o
def get_database_connection(database_name='mysql'):
    try:
        return mdb.connect(host=HOST, user=USER, passwd=PASSWORD, db=database_name)
    except Exception as e:
        logger.warning('Failed to establish databse connection: ' + str(e))
        return None


def execute_sql_input_script(database_name, scripts):
    db = get_database_connection(database_name)
    if db is not None:
        try:
            cur = db.cursor()
            cur.execute(scripts)
            db.commit()
            db.close()
            return None
        except Exception as e:
            db.close()
            return e


def execute_sql_output_script(database_name, scripts):
    db = get_database_connection(database_name)
    if db is not None:
        try:
            cur = db.cursor()
            cur.execute(scripts)
            data = cur.fetchall()
            db.close()
            return True, data
        except Exception as e:
            db.close()
            return False, e


def create_table(database_name, table_name, table_format):
    script = CREATE_TABLE_IF_NOT_EXISTS % (table_name, table_format)
    e = execute_sql_input_script(database_name, script)
    if e is not None:
        logger.warning('Failed to create table: ' + str(e))


def create_timeseries_table(database_name, table_name):
    create_table(database_name, table_name, TIMESERIES_TABLE_FORMAT)


def create_t2_timeseries_table(database_name, table_name):
    create_table(database_name, table_name, T2_TIMESERIES_TABLE_FORMAT)


def create_description_table(database_name, table_name):
    create_table(database_name, table_name, DESCRIPTION_TABLE_FORMAT)


def pandas_bulk_insert(data, database_name, table_name, column_name, index_name, value_name, data_name=None, data_column_name=None):
    delete_script = get_pandas_bulk_delete_script(data, table_name, column_name, index_name, data_name, data_column_name)
    e = execute_sql_input_script(database_name, delete_script)
    if e is not None:
        logger.warning('Failed to clear data from table: ' + str(e))
    else:
        insert_script = get_pandas_bulk_insert_script(data, table_name, column_name, index_name, value_name, data_name, data_column_name)
        e = execute_sql_input_script(database_name, insert_script)
        if e is not None:
            logger.warning('Failed to insert data: ' + str(e))


def get_pandas_output(data, column_name, index_name, value_name):
    output = pd.DataFrame(np.array(data), columns=[column_name, index_name, value_name])
    return output.pivot(index_name, column_name, value_name).sort_index().fillna(np.nan)


def pandas_read(database_name, table_name, column_name, index_name, value_name, index_range=None, column_list=None, data_name=None):
    read_script = get_pandas_read_script(table_name, column_name, index_name, value_name, index_range, column_list, data_name)
    success, data = execute_sql_output_script(database_name, read_script)
    if success:
        return get_pandas_output(data, column_name, index_name, value_name) if len(data)>0 else None
    else:
        logger.warning('Failed to read data: ' + str(data))


def pandas_delete(database_name, table_name, column_name, index_name, value_name, index_range=None, column_list=None, data_name=None):
    delete_script = get_pandas_delete_script(table_name, column_name, index_name, value_name, index_range, column_list, data_name)
    e = execute_sql_input_script(database_name, delete_script)
    if e is not None:
        logger.warning('Failed to delete data from table: ' + str(e))


def get_table_column_values(database_name, table_name, column_name=TIMESERIES_COLUMN_NAME):
    read_script = get_table_column_value_scripts(table_name, column_name)
    success, data = execute_sql_output_script(database_name, read_script)
    if success:
        return [x[0] for x in data] if len(data)>0 else None
    else:
        logger.warning('Failed to read column names: ' + str(data))

    
