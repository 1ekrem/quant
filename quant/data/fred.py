'''
Created on 6 Jul 2017

@author: wayne
'''
import sys
import fredapi
import numpy as np
import pandas as pd
from datetime import timedelta
from quant.lib import data_utils as du, timeseries_utils as tu
from quant.lib.main_utils import logger

DATABASE_NAME = 'quant'
INFO_TABLE_NAME = 'fred_econ_info'
SERIES_TABLE_NAME = 'fred_data'
RELEASE_TABLE_NAME = 'fred_econ_data'
CACHE_TABLE_NAME = 'fred_cached_data'
TABLE_FORMAT = "time_index DATETIME, realtime_start DATETIME, series_name VARCHAR(50), value DOUBLE"
INFO_TABLE_FORMAT = "series_name VARCHAR(50), description VARCHAR(50), value VARCHAR(1000)"
FREDKEY = 'ff64294203f79127f8d004d2726386ac'
_api = fredapi.Fred(api_key=FREDKEY)

# data config
US_ECON = [# Economic indicator
           'PAYEMS', 'USSLIND', 'FRBLMCI',
           # Inflation
           'T10YIE', 'T5YIFR', 'MICH', 'CPILFESL', 'PPIACO',
           # Consumption
           'UMCSENT', 'PCEC96', 'TOTALSA', 'RSXFS',
           # National income
           'GDP', 'GDPC1',
           # Interest rates
           'WGS10YR', 'WGS5YR', 'WGS2YR', 'WGS1YR', 'MORTGAGE30US', 'FF', 'DTB3', 'FEDFUNDS', 'DFEDTARU',
           'T10Y2Y', 'T10Y3M', 'BAMLH0A0HYM2EY', 'TEDRATE',
           # Current population survey
           'UNRATE', 'IC4WSA',
           # Housing
           'HOUST', 'HOUST1F', 'CSUSHPINSA', 'MSPNHSUS', 'HSN1F',
           # Industrial and manufacturing
           'INDPRO', 'DGORDER', 'NEWORDER',
           # Transportation
           'RAILFRTINTERMODAL',
           # Corporate bond yield
           'WAAA', 'WBAA', 'BAMLC0A4CBBBEY',
           # Risk indicator
           'DRSFRMACBS', 'BAMLH0A0HYM2', 'BAMLC0A4CBBB', 'BAMLH0A3HYC', 'DRCCLACBS', 'USREC', 'USARECM',
           ]
EU_ECON = [# Interest rates
           'IRLTLT01DEM156N', 'BAMLHE00EHYIEY', 'BAMLHE00EHYIOAS', 'EUR3MTD156N',
           'BAMLEMRECRPIEMEAOAS', 'EUR12MD156N', 'BAMLEMRECRPIEMEAEY', 'INTGSBEZM193N',
           # Economic indicator
           'EUEPUINDXM', 'EUNNGDP', 'SLRTTO01OEQ659S', 'CLVMEURSCAB1GQEA19',
           'SLRTCR03OEQ661S', 'CRDQXMAPABIS',
           # Inflation
           'CP0000EZ19M086NEST', 'CPALTT01OEM661N',
           # Risk indicators
           'BAMLHE00EHYITRIV', 'EUROREC', 'EUEPUINDXM',
           ]
CHINA_ECON = [# Economic indicator
              'CHNCPIALLMINMEI', 'MKTGDPCNA646NWDB', 'CHNGDPNQDSMEI', 'CRDQCNAPABIS',
              'XTEXVA01CNM667S', 'XTIMVA01CNM667S', 'SLRTTO02CNQ189N', 'PRENEL01CNQ656N',
              # Monetary policy
              'MYAGM2CNM189N', 'MANMM101CNM189S', 'MABMM301CNM189S',
              # Inflation
              'CHNCPIALLMINMEI', 'FPCPITOTLZGCHN',
              # Real estate
              'QCNR368BIS',
              # Interest rate
              'INTDSRCNM193N',
              # Risk indicators
              'VXFXICLS', 'CHIEPUINDXM', 'CHNRECM',
              ]
US_SERIES = [# Stock
             'SP500', 'NASDAQCOM', 'DJIA', 'VIXCLS',
             # FX
             'DEXUSEU', 'DEXUSUK', 'DEXUSNZ', 'DEXUSAL', 'DEXJPUS',
             # Commodities
             'DCOILWTICO', 'DCOILBRENTEU', 'GOLDAMGBD228NLBM',
             # Interest rates
             'USD3MTD156N', 'USD12MD156N', 'USDONTD156N',
             ]

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
    delete_format_script = "(time_index, realtime_start, series_name)"
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


def calculate_first_release(data):
    if data is not None:
        ans = data.groupby('realtime_start').agg(lambda x:x.sort_values('time_index').iloc[-1]).sort_index()['value']
        return None if ans.empty else tu.remove_outliers(ans)
    else:
        return None


def calculate_extended_first_release(data):
    if data is not None:
        data = data.sort_values(['time_index', 'realtime_start'])
        current = data.groupby('realtime_start').agg(lambda x:x.sort_values('time_index').iloc[-1])
        ans = current['value'].copy()
        current = current.reset_index()
        time_delta = (current['realtime_start'] - current['time_index']).mean().days
        historic = data.groupby('time_index').agg(lambda x:x.sort_values('realtime_start').iloc[0])
        historic = historic.loc[historic.index < current.time_index.min()]
        ans2 = historic['value'].copy()
        ans2.index += timedelta(time_delta)
        ans = pd.concat([ans2, ans], axis=0)
        return None if ans.empty else tu.remove_outliers(ans)
    else:
        return None


def calculate_change(data, time_delta):
    '''
    time_delta is calendar days
    '''
    if data is not None:
        data = data.sort_values(['time_index', 'realtime_start'])
        release = data.groupby('realtime_start').agg(lambda x:x.sort_values('time_index').iloc[-1]).reset_index()
        history = pd.Series()
        for idx in release.time_index:
            subset = data[data.time_index == idx]
            effective_date = subset.realtime_start.min() + timedelta(time_delta)
            subset2 = subset[subset.realtime_start >= effective_date]
            history.loc[effective_date] = subset.value.iloc[-1] if subset2.empty else subset2.value.iloc[0]
        release.index = release.realtime_start
        history = tu.resample(history, release)
        ans = release.value - history
        td = (release['realtime_start'] - release['time_index']).mean().days
        historic = data.groupby('time_index').agg(lambda x:x.sort_values('realtime_start').iloc[0])
        historic = historic.loc[historic.index < release.time_index.min()]
        observation = historic.value
        pasttime = observation.copy()
        pasttime.index -= timedelta(time_delta)
        ans2 = pasttime - tu.resample(observation, pasttime)
        ans2.index = observation.index + timedelta(td)
        ans = pd.concat([ans2, ans], axis=0)
        return None if ans.empty else tu.remove_outliers(ans)
    else:
        return None


def calculate_revision(data):
    if data is not None:
        data = data.sort_values(['time_index', 'realtime_start'])
        data['diff'] = data.value.diff()
        data.loc[data.time_index != data.time_index.shift(), 'diff'] = np.nan
        data = data.dropna()
        data = data.groupby('realtime_start').agg(lambda x:x.sort_values('time_index').iloc[-1]).reset_index()
        data = data.groupby('time_index').first().reset_index()
        ans = data['diff']
        ans.index = data.realtime_start
        return None if ans.empty else tu.remove_outliers(ans)
    else:
        return None


def cache_series_release_data(series_name):
    logger.info('Caching derived series release data - %s' % series_name)
    data = get_series_all_release(series_name)
    if data is not None:
        ans = []
        release = calculate_first_release(data)
        if release is not None:
            release.name = series_name + '|release'
            ans.append(release)
        release = calculate_extended_first_release(data)
        if release is not None:
            release.name = series_name + '|extendedrelease'
            ans.append(release)
        change = calculate_change(data, 5)
        if change is not None:
            change.name = series_name + '|change'
            ans.append(change)
        change = calculate_change(data, 365)
        if change is not None:
            change.name = series_name + '|annualchange'
            ans.append(change)
        revision = calculate_revision(data)
        if revision is not None:
            revision.name = series_name + '|revision'
            ans.append(revision)
        if len(ans)>0:
            ans = pd.concat(ans, axis=1)
            tu.store_timeseries(ans, DATABASE_NAME, CACHE_TABLE_NAME)


def download_all_releases():
    for series_name in US_ECON + CHINA_ECON + EU_ECON:
        download_and_store_series_all_releases(series_name)
        cache_series_release_data(series_name)


def download_all_series():
    for series_name in US_SERIES:
        download_and_store_series(series_name)


# data loader
def get_fred_us_econ_list():
    return US_ECON


def get_fred_global_econ_list():
    return US_ECON + CHINA_ECON + EU_ECON


def get_cached_data(series_name, data_type, start_date=None, end_date=None):
    column = series_name + '|' + data_type
    ans = tu.get_timeseries(DATABASE_NAME, CACHE_TABLE_NAME, index_range=(start_date, end_date), column_list=[column])
    if ans is None:
        return None
    else:
        ans = ans.iloc[:, 0]
        ans.name = series_name
        return ans


def get_fred_first_release(series_name, start_date=None, end_date=None):
    return get_cached_data(series_name, 'release', start_date, end_date)


def get_fred_extended_first_release(series_name, start_date=None, end_date=None):
    return get_cached_data(series_name, 'extendedrelease', start_date, end_date)
    

def get_fred_change(series_name, start_date=None, end_date=None):
    return get_cached_data(series_name, 'change', start_date, end_date)


def get_fred_annual_change(series_name, start_date=None, end_date=None):
    return get_cached_data(series_name, 'annualchange', start_date, end_date)


def get_fred_revision(series_name, start_date=None, end_date=None):
    return get_cached_data(series_name, 'revision', start_date, end_date)


def get_fred_combined(series_name, start_date=None, end_date=None):
    ans = []
    release = get_fred_extended_first_release(series_name, start_date, end_date)
    if release is not None:
        release.name = series_name + '|release'
        ans.append(release)
    change = get_fred_change(series_name, start_date, end_date)
    if change is not None:
        change.name = series_name + '|change'
        ans.append(change)
    change = get_fred_annual_change(series_name, start_date, end_date)
    if change is not None:
        change.name = series_name + '|annualchange'
        ans.append(change)
    revision = get_fred_revision(series_name, start_date, end_date)
    if revision is not None:
        revision.name = series_name + '|revision'
        ans.append(revision)
    return None if len(ans) == 0 else pd.concat(ans, axis=1)


def fred_release_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, get_fred_extended_first_release(ticker, start_date, end_date)) for ticker in tickers])


def fred_change_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, get_fred_change(ticker, start_date, end_date)) for ticker in tickers])


def fred_annual_change_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, get_fred_annual_change(ticker, start_date, end_date)) for ticker in tickers])


def fred_revision_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, get_fred_revision(ticker, start_date, end_date)) for ticker in tickers])


def fred_combined_loader(tickers, start_date=None, end_date=None):
    return dict([(ticker, get_fred_combined(ticker, start_date, end_date)) for ticker in tickers])


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
