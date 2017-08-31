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
US_ECON = [  # Economic indicator
           'PAYEMS', 'USSLIND', 'FRBLMCI',
           # Inflation
           'T10YIE', 'T5YIFR', 'MICH', 'CPILFESL', 'PPIACO', 'CPIAUCSL', 'CPALTT01USQ661S', 'CPALTT01USM661S',
           'CPALTT01USM659N',
           # Consumption
           'UMCSENT', 'PCEC96', 'TOTALSA', 'RSXFS',
           # Auto
           'M12MTVUSM227SFWA', 'AISRSA', 'DAUPSA', 'VMTD11', 'DAUTOSAAR', 'IPG3361T3S', 'LAUTOSA',
           'AUESA', 'FAUTOSA', 'MAUISA', 'CAUISA', 'B149RC1Q027SBEA',
           # National income
           'GDP', 'GDPC1',
           # Interest rates
           'WGS10YR', 'WGS5YR', 'WGS2YR', 'WGS1YR', 'MORTGAGE30US', 'FF', 'DTB3', 'FEDFUNDS', 'DFEDTARU',
           'T10Y2Y', 'T10Y3M', 'BAMLH0A0HYM2EY', 'TEDRATE',
           # Current population survey
           'UNRATE', 'IC4WSA', 'U6RATE', 'LNS14027662', 'CCSA', 'UEMPMEAN', 'IURSA',
           # Housing
           'HOUST', 'HOUST1F', 'CSUSHPINSA', 'MSPNHSUS', 'HSN1F', 'DRCRELEXFACBS', 'CSUSHPISA', 'SPCS20RSA',
           'PERMIT', 'RSAHORUSQ156S', 'MSPUS', 'COMPUTNSA',
           # Industrial and manufacturing
           'INDPRO', 'DGORDER', 'NEWORDER',
           # Transportation
           'RAILFRTINTERMODAL',
           # Corporate bond yield
           'WAAA', 'WBAA', 'BAMLC0A4CBBBEY',
           # Risk indicator
           'DRSFRMACBS', 'BAMLH0A0HYM2', 'BAMLC0A4CBBB', 'BAMLH0A3HYC', 'DRCCLACBS', 'USREC', 'USARECM', 'DRSREACBS',
           # QE
           'WSHOTS', 'WSHOMCB',
           ]
EU_ECON = [  # Interest rates
           'IRLTLT01DEM156N', 'BAMLHE00EHYIEY', 'BAMLHE00EHYIOAS', 'EUR3MTD156N', 'INTDSREZM193N',
           'BAMLEMRECRPIEMEAOAS', 'EUR12MD156N', 'BAMLEMRECRPIEMEAEY', 'INTGSBEZM193N',
           # Unemployment
           'LMUNRRTTDEM156S', 'LRUN74TTFRQ156N', 'LRHUTTTTITM156S', 'LRHUTTTTESM156S',
           # Economic indicator
           'EUNNGDP', 'SLRTTO01OEQ659S', 'CLVMEURSCAB1GQEA19', 'CLVMNACSCAB1GQEU15',
           'CRDQXMAPABIS',
           # Real estate
           'QESR628BIS', 'QDER628BIS', 'QESR368BIS', 'QDER368BIS', 'ODCNPI03DEQ180S', 'ODCNPI03DEQ659S',
           'QISR628BIS', 'ODCNPI03ESQ470S', 'ODCNPI03ESQ659S',
           # Auto
           'SLRTCR03FRQ657S', 'SLRTCR03DEQ657S', 'SLRTCR03HUQ657S', 'SLRTCR03ITQ657S', 'SLRTCR03DKQ657S',
           'SLRTCR03OEQ657S', 'CP0710EZ18M086NEST', 'CP0711EZ18M086NEST', 'SLRTCR03OEQ661S',
           'SLRTCR03FRQ661S', 'SLRTCR03DEQ661S', 'SLRTCR03ESQ180S', 'SLRTCR03ESQ657S', 'SLRTCR03ITQ661S',
           'SLRTCR03DKQ180S',
           # Inflation
           'CP0000EZ19M086NEST', 'CPALTT01OEM661N', 'FPCPITOTLZGEUU', 'DEUCPIALLMINMEI',
           # Risk indicators
           'BAMLHE00EHYITRIV', 'EUROREC', 'EUEPUINDXM',
           # QE
           'ECBASSETS', 'RBXMBIS',
           ]
CHINA_ECON = [  # Economic indicator
              'MKTGDPCNA646NWDB', 'CHNGDPNQDSMEI', 'CRDQCNAPABIS',
              'XTEXVA01CNM667S', 'XTIMVA01CNM667S', 'SLRTTO02CNQ189N', 'PRENEL01CNQ656N',
              # Monetary policy
              'MYAGM2CNM189N', 'MANMM101CNM189S', 'MABMM301CNM189S',
              # Inflation
              'CHNCPIALLMINMEI', 'FPCPITOTLZGCHN',
              # Real estate
              'QCNR368BIS', 'QHKR368BIS', 'QCNN368BIS',
              # Interest rate
              'INTDSRCNM193N', 'QHKR628BIS', 'QCNN628BIS',
              # Risk indicators
              'VXFXICLS', 'CHIEPUINDXM', 'CHNRECM',
              ]
UK_ECON = [  # Economic indicators
           'CLVMNACSCAB1GQUK', 'LORSGPORGBQ659S', 'CPALTT01GBM657N',
           # Inflation
           'GBRCPIALLMINMEI',
           # Real estate
           'QGBN628BIS', 'QCNR628BIS', 'QGBR628BIS', 'QGBR368BIS',
           # Unemployment
           'LMUNRRTTGBM156S', 'AURUKM',
           # Auto
           'SLRTCR03GBQ657S', 'SLRTCR03GBQ180N', 'SLRTCR03GBQ180S', 'CP0710GBM086NEST',
          ]
EM_ECON = [  # Risk indicators
           'BAMLEMCBPIOAS', 'BAMLEMHBHYCRPIOAS', 'BAMLEM3BRRBBCRPIOAS',
           # Real estate
           'Q4TR628BIS', 'Q4TR771BIS', 'QBRR628BIS',
           # Inflation
           'INDCPIALLMINMEI', 'BRACPIALLMINMEI', 'KORCPIALLMINMEI',
           ]
ROW_ECON = [# Unemployment
            'LRUNTTTTCAM156S', 'LRUN64TTJPM156S', 'LRHUTTTTJPM156S', 'LRHUTTTTCAM156S', 'LRHUTTTTAUM156S',
            # Real estate
            'QMYR628BIS', 'QCAR628BIS', 'QMYR368BIS', 'QCAR368BIS', 'ODCNPI03AUQ189S', 'ODCNPI03AUA156N',
            'ODCNPI03AUQ657S', 'ODCNPI03AUQ659S', 'QKRR628BIS', 'AUSPERMITMISMEI', 'QKRR368BIS',
            'QAUN628BIS', 'QAUR628BIS', 'QIDR368BIS', 'QCAN628BIS', 'QSGN628BIS', 'QMXN628BIS',
            'QRUN628BIS',
            # Inflation
            'JPNCPIALLMINMEI', 'CPALCY01CAM661N', 'PALLFNFINDEXM', 'CPGRLE01JPM657N', 'AUSCPIALLQINMEI',
            'MEXCPIALLMINMEI',
            # Auto
            'SLRTCR03AUQ659S', 'SLRTCR03AUQ180S', 'SLRTCR03AUQ657S', 'SLRTCR03JPQ657S', 'SLRTCR03JPQ180S',
            ]
US_SERIES = [  # Stock
             'SP500', 'NASDAQCOM', 'DJIA', 'VIXCLS',
             # FX
             'DEXUSEU', 'DEXUSUK', 'DEXUSNZ', 'DEXUSAL', 'DEXJPUS',
             # Commodities
             'DCOILWTICO', 'DCOILBRENTEU', 'GOLDAMGBD228NLBM',
             # Interest rates
             'USD3MTD156N', 'USD12MD156N', 'USDONTD156N',
             ]

# utils
def find_series(series_id):
    ids = get_fred_global_econ_list()
    if series_id in ids:
        print(get_series_info(series_id))


def check_series():
    series = get_fred_global_econ_list()
    if len(series) == len(set(series)):
        print('Passed')
    else:
        tmp = sorted(series)
        for i in xrange(len(tmp) - 1):
            if tmp[i] == tmp[i + 1]:
                print(tmp[i])


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
        return pd.DataFrame(np.array(data), columns=['time_index', 'realtime_start', 'series_name', 'value']) if len(data) > 0 else None
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
        if len(ans) > 0:
            ans = pd.concat(ans, axis=1)
            logger.info('Storing derived series release data - %s' % series_name)
            tu.store_timeseries(ans, DATABASE_NAME, CACHE_TABLE_NAME)


def download_all_releases():
    for series_name in get_fred_global_econ_list():
        download_and_store_series_all_releases(series_name)
        cache_series_release_data(series_name)


def download_all_series():
    for series_name in US_SERIES:
        download_and_store_series(series_name)


# data loader
def get_fred_us_econ_list():
    return US_ECON


def get_fred_us_eu_econ_list():
    return US_ECON + EU_ECON


def get_fred_global_econ_list():
    return US_ECON + CHINA_ECON + EU_ECON + UK_ECON + EM_ECON + ROW_ECON


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
