'''
Created on 22 Jun 2017

@author: wayne
'''
import pandas as pd
from datetime import datetime as dt
from quant.lib import timeseries_utils as tu, data_utils as du, portfolio_utils as pu
from quant.lib.main_utils import logger

DATABASE_NAME = 'quant'
INDEX_TABLE_NAME = 'bloomberg_index_prices'
US_ECON_TABLE_NAME = 'bloomberg_us_econ'
ACTUAL_RELEASE = 'ACTUAL_RELEASE'
PX_LAST = 'PX_LAST'
DATA_MISSING_FAIL = 0.7


def load_bloomberg_index_prices(ticker='SPX Index'):
    return tu.get_timeseries(DATABASE_NAME, INDEX_TABLE_NAME, column_list=[ticker])


def load_bloomberg_econ_release(ticker, table_name=US_ECON_TABLE_NAME):
    data = tu.get_timeseries(DATABASE_NAME, table_name, column_list=[ticker + '|' + ACTUAL_RELEASE])
    if data is not None:
        data = data.iloc[:, 0]
        data.name = ticker
    return data


def load_bloomberg_econ_last(ticker, table_name=US_ECON_TABLE_NAME):
    data = tu.get_timeseries(DATABASE_NAME, table_name, column_list=[ticker + '|' + PX_LAST])
    if data is not None:
        data = data.iloc[:, 0]
        data.name = ticker
    return data


def load_bloomberg_econ_change(ticker, table_name=US_ECON_TABLE_NAME):
    release = load_bloomberg_econ_release(ticker, table_name)
    last = load_bloomberg_econ_last(ticker, table_name)
    if release is not None and last is not None:
        return release - last.shift()
    else:
        return None


def load_bloomberg_econ_revision(ticker, table_name=US_ECON_TABLE_NAME):
    release = load_bloomberg_econ_release(ticker, table_name)
    last = load_bloomberg_econ_last(ticker, table_name)
    if release is not None and last is not None:
        return (last - release).shift()
    else:
        return None


def get_bloomberg_econ_list(table_name=US_ECON_TABLE_NAME):
    vals = du.get_table_column_values(DATABASE_NAME, table_name)
    return vals if vals is None else list(set([x.split('|')[0] for x in vals]))


# Simulations
class USEconBoosting(object):
    '''
    Univariate forecasting
        -    Original data vs. Score
        -    Raw, change, revision
    '''
    def __init__(self, assets, start_date, end_date, frequency, sample_window,
                 model_frequency, table_name=INDEX_TABLE_NAME, data_missing_fail=DATA_MISSING_FAIL,
                 *args, **kwargs):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.sample_window = sample_window
        self.model_frequency = model_frequency
        self.table_name = table_name
        self.data_missing_fail = data_missing_fail
        self.run_simulation()

    def run_simulation(self):
        self.get_timeline()
        self.load_asset_prices()
        self.load_economic_datasets()
        self.run_univariate_experiments()

    def get_timeline(self):
        logger.info('Creating time line')
        self.timeline = pu.get_timeline(self.start_date, self.end_date, self.frequency, self.sample_window)
        self.model_timeline = self.timeline.resample(self.model_frequency).last()
        self._load_start = self.timeline.index[0]
        self._load_end = self.timeline.index[-1]

    def load_asset_prices(self):
        logger.info('Loading asset prices')
        self.asset_prices = tu.get_timeseries(DATABASE_NAME, self.table_name, (self._load_start, self._load_end), self.assets)

    def load_economic_datasets(self):
        logger.info('Loading economic variables')
        self.economic_variables = get_bloomberg_econ_list(US_ECON_TABLE_NAME)
        self.economic_dataset = {}
        logger.info('Loading economic release')
        release = []
        for ticker in self.economic_variables:
            data = load_bloomberg_econ_release(ticker)
            if data is not None:
                release.append(tu.resample(data, self.timeline))
        self.economic_dataset['Release'] = pd.concat(release, axis=1)
        logger.info('Loading economic change')
        change = []
        for ticker in self.economic_variables:
            data = load_bloomberg_econ_change(ticker)
            if data is not None:
                change.append(tu.resample(data, self.timeline))
        self.economic_dataset['Change'] = pd.concat(change, axis=1)
        logger.info('Loading economic revision')
        revision = []
        for ticker in self.economic_variables:
            data = load_bloomberg_econ_revision(ticker)
            if data is not None:
                revision.append(tu.resample(data, self.timeline))
        self.economic_dataset['Revision'] = pd.concat(revision, axis=1)
    
    def run_univariate_experiments(self):
        for data_type in ['Release', 'Change', 'Revision']:
            dataset = self.economic_dataset[data_type]
            for ticker in self.economic_variables:
                data = tu.resample(dataset[ticker], self.timeline).to_frame()
                for input_type in ['Original', 'Score']:
                    models = []
                    for idx, model_time in enumerate(self.model_timeline.index):
                        logger.info('Running %s %s %s at %s' % (ticker, data_type, input_type, model_time.strftime('%Y-%m-%d')))
                        in_sample = self.timeline.copy()
                        in_sample = in_sample.loc[in_sample.index <= model_time]
                        in_sample = in_sample.iloc[-self.sample_window:]
                        in_sample_data = pu.ignore_insufficient_series(data.loc[in_sample.index],
                                                                       self.sample_window * self.data_missing_fail)
                        if in_sample_data is None:
                            logger.info('Insufficient Data - ignored')
                        else:
                            out_of_sample = self.timeline.copy()
                            out_of_sample = out_of_sample.loc[out_of_sample.index > model_time]
                            if idx < len(self.model_timeline) - 1:
                                next_date = self.model_timeline.index[idx + 1]
                                out_of_sample = out_of_sample.loc[out_of_sample.index <= next_date]
                            out_of_sample_data = data.loc[out_of_sample.index]
                            
                            
                        
                

def run_us_econ_boosting():
    sim = USEconBoosting(['SPX Index'], dt(2002, 1, 1), dt(2017, 6, 1), 'M', 24, 'Q')
    return sim
    