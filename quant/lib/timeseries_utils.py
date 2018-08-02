'''
Created on Jun 1, 2017

@author: Wayne
'''
import pandas as pd
import numpy as np
from scipy import stats as ss
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


def remove_outliers(data, z=15):
    x = data / data.std()
    ans = data.copy()
    ans[x.abs()>=z] = np.nan
    return ans


def get_clean_returns(r, threshold=4.):
    ans = r.copy()
    ans[r > threshold] = threshold
    ans[r < -threshold] = -threshold
    return ans
    
    
def ts_interpolate(series):
    '''
    Create a zero observation when timeseries crosses the time axis
    '''
    assert isinstance(series, pd.Series)
    data = series.dropna()
    a = data.abs()
    df = np.diff(data.index.values)
    s = a.rolling(2).sum()
    s[s<=0.] = np.nan
    fr = a.values[:-1] / s.values[1:]
    fr[np.isnan(fr)] = .5
    check = data.values[:-1] * data.values[1:]
    idx = fr * df + data.index.values[:-1]
    idx = idx[check < 0]
    if len(idx) > 0:
        new_series = pd.Series([0.] * len(idx), index=idx, name=series.name)
        data = pd.concat([data, new_series], axis=0).sort_index()
    return data


def store_timeseries(ts, database_name, table_name, data_name=None):
    du.pandas_bulk_insert(ts, database_name, table_name, du.TIMESERIES_COLUMN_NAME, du.TIMESERIES_INDEX_NAME,
                          du.TIMESERIES_VALUE_NAME, data_name, du.TIMESERIES_DATA_NAME)


def store_description(des, database_name, table_name):
    du.pandas_bulk_insert(des, database_name, table_name, du.DESCRIPTION_COLUMN_NAME, du.DESCRIPTION_INDEX_NAME,
                          du.DESCRIPTION_VALUE_NAME)
    

def delete_timeseries(database_name, table_name, index_range=None, column_list=None, data_name=None):
    du.pandas_delete(database_name, table_name, du.TIMESERIES_COLUMN_NAME, du.TIMESERIES_INDEX_NAME, du.TIMESERIES_VALUE_NAME, index_range, column_list, data_name)


def get_timeseries(database_name, table_name, index_range=None, column_list=None, data_name=None):
    return du.pandas_read(database_name, table_name, du.TIMESERIES_COLUMN_NAME, du.TIMESERIES_INDEX_NAME, du.TIMESERIES_VALUE_NAME, index_range, column_list, data_name)


def get_description(database_name, table_name, column_list=None):
    return du.pandas_read(database_name, table_name, du.DESCRIPTION_COLUMN_NAME, du.DESCRIPTION_INDEX_NAME, du.DESCRIPTION_VALUE_NAME, None, column_list, None)    


def delete_description(database_name, table_name, column_list=None):
    du.pandas_delete(database_name, table_name, du.DESCRIPTION_COLUMN_NAME, du.DESCRIPTION_INDEX_NAME, du.DESCRIPTION_VALUE_NAME, None, column_list, None)
    

def get_distribution_parameters(data):
    '''
    Mean, std and medians of series
    '''
    assert isinstance(data, pd.DataFrame)
    ans = pd.concat([data.mean(axis=0), data.median(axis=0), data.std(axis=0)], axis=1)
    ans.columns = ['mean', 'median', 'std']
    return ans


def get_distribution_scores(data, params):
    '''
    Converts data into distribution scores
    '''
    ans = (data - params['mean']) / params['std']
    ans.loc[:, :] = ss.norm.cdf(ans.values)
    ans[data.isnull()] = np.nan
    return ans


def pandas_ewma(data, span=None, *args, **kwargs):
    return data.ewm(span=span, ignore_na=True).mean() if span > 1. else data


def dataframe_to_series(data):
    return pd.Series(data.values.flatten())


def series_to_dataframe(data, columns, indices):
    return pd.DataFrame(np.reshape(data.values.flatten(), (len(indices), len(columns))), index=indices, columns=columns)
