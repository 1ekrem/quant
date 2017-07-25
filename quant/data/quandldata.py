'''
Created on 25 Jul 2017

@author: wayne
'''
QUANDL_KEY = "uwzfxL-Kxg5sMVfVuHSW"

import numpy as np
import quandl as qd
from quant.lib import data_utils as du
from quant.lib.main_utils import logger

qd.ApiConfig.api_key = QUANDL_KEY
DATABASE_NAME = 'quant'
QUANDL_TABLE_NAME = 'quandl_data'
TABLE_FORMAT = "time_index DATETIME, series_name VARCHAR(50), series_type VARCHAR(50), value DOUBLE"


def create_table():
    du.create_table(DATABASE_NAME, QUANDL_TABLE_NAME, TABLE_FORMAT)


def load_series(series_id):
    try:
        return qd.get(series_id)
    except Exception as e:
        logger.warn('Failed to load quandl data %s: %s' % (series_id, str(e)))
        return None
