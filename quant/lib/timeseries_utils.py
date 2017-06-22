'''
Created on Jun 1, 2017

@author: Wayne
'''
import pandas as pd
import numpy as np
from quant.lib import data_utils as du



def resample(ts, timeline, carry_forward=True):
    assert isinstance(ts, pd.Series) or isinstance(ts, pd.DataFrame)
    assert isinstance(timeline, pd.Series) or isinstance(timeline, pd.DataFrame)
    index = set(ts.index)
    idx = set([x for x in timeline.index if x not in index])
    if len(idx)>0:
        if isinstance(ts, pd.Series):
            tmp = pd.Series([np.nan] * len(idx), index=idx, name=ts.name)
        else:
            tmp = pd.DataFrame(np.ones((len(idx), np.size(ts,1))) * np.nan, index=idx, columns=ts.columns)
        ts = pd.concat([ts, tmp], axis=0).sort_index()
        if carry_forward:
            for t in range(len(ts)-1):
                if ts.index[t+1] in idx:
                    ts.iloc[t+1] = ts.iloc[t]
    return ts.loc[timeline.index]


def store_timeseries(ts, database_name, table_name):
    du.pandas_bulk_insert(ts, database_name, table_name, du.TIMESERIES_COLUMN_NAME, du.TIMESERIES_INDEX_NAME, du.TIMESERIES_VALUE_NAME)


def get_timeseries(database_name, table_name, index_range=None, column_list=None):
    return du.pandas_read(database_name, table_name, du.TIMESERIES_COLUMN_NAME, du.TIMESERIES_INDEX_NAME, du.TIMESERIES_VALUE_NAME, index_range, column_list)
