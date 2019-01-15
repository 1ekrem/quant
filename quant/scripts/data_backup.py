from quant.lib.main_utils import *
import pandas as pd
from quant.lib import timeseries_utils as tu
from quant.data import stocks


def get_filename(data_type):
    return os.path.expanduser('~/%s_backup.xlsx' % data_type)


def export_lse_spreads():
    filename = get_filename('spread')
    data = stocks.load_google_returns(data_name='Spread', data_table=stocks.UK_STOCKS)
    f = pd.ExcelWriter(filename)
    data.to_excel(f)
    f.save()


def import_lse_spreads():
    filename = get_filename('spread')
    ans = pd.read_excel(filename)
    ans.index = ans.time_index
    ans = ans.iloc[:, 1:]
    tu.store_timeseries(ans, stocks.DATABASE_NAME, stocks.UK_STOCKS, 'Spread')


def export_lse_ids():
    filename = get_filename('LSEID')
    data = stocks.get_universe('LSE')
    f = pd.ExcelWriter(filename)
    data.to_excel(f)
    f.save()


def import_lse_ids():
    filename = get_filename('LSEID')
    ans = pd.read_excel(filename)
    ans.index = ans.column_name
    stocks._save_tickers(ans.LSE, 'LSE')


def export_reuters_data():
    tags = ['Rating', 'C1', 'C3', 'T1', 'T4', 'T8', 'T52']
    filename = get_filename('Reuters')
    f = pd.ExcelWriter(filename)
    for data_name in tags:
        data = stocks.load_google_returns(data_name=data_name, data_table=stocks.UK_ESTIMATES)
        data.to_excel(f, sheet_name=data_name)
    f.save()


def import_reuters_data():
    tags = ['Rating', 'C1', 'C3', 'T1', 'T4', 'T8', 'T52']
    filename = get_filename('Reuters')
    f = pd.ExcelFile(filename)
    for data_name in tags:
        data = f.parse(data_name)
        data.index = data.time_index
        data = data.iloc[:, 1:]
        tu.store_timeseries(data, stocks.DATABASE_NAME, stocks.UK_ESTIMATES, data_name)


def export_reuters_ids():
    filename = get_filename('ReutersID')
    data = stocks.get_universe('Reuters')
    f = pd.ExcelWriter(filename)
    data.to_excel(f)
    f.save()


def import_reuters_ids():
    filename = get_filename('ReutersID')
    ans = pd.read_excel(filename)
    ans.index = ans.column_name
    stocks._save_tickers(ans.Reuters, 'Reuters')


def export_stock_returns():
    filename = get_filename('Returns')
    f = pd.ExcelWriter(filename)
    for data_type in ['Returns', 'Volume']:
        data = stocks.load_google_returns(data_name=data_type, data_table=stocks.UK_STOCKS)
        data.to_excel(f, sheet_name=data_type)
    f.save()


def import_stock_returns():
    filename = get_filename('Returns')
    f = pd.ExcelFile(filename)
    for data_type in ['Returns', 'Volume']:
        ans = f.parse(data_type)
        ans.index = ans.time_index
        ans = ans.iloc[:, 1:]
        for c in ans.columns:
            logger.info('Storing %s' % c)
            tu.store_timeseries(ans.loc[:, c].dropna().to_frame(), stocks.DATABASE_NAME, stocks.UK_STOCKS, data_type)


def export_data():
    export_lse_spreads()
    export_lse_ids()
    export_reuters_data()
    export_reuters_ids()
    export_stock_returns()


def import_data():
    import_lse_spreads()
    import_lse_ids()
    import_reuters_data()
    import_reuters_ids()
    import_stock_returns()

