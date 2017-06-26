'''
Created on 25 Jun 2017

@author: wayne
'''

import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta


def get_timeline(start_date, end_date, frequency, sample_window=None):
    '''
    Create a timeline for simulation

    Input
    --------
    start_date       start date of simulation 
    end_date         end date of simulation
    frequency        'B', '1-5', 'M', 'MS', 'Q', 'A'
    sample_window    integer of the number of observations for sample size of estimation
    '''
    assert isinstance(start_date, dt)
    assert isinstance(end_date, dt)
    assert frequency in ['B', '1', '2', '3', '4', '5', 'M', 'MS', 'Q', 'A']
    if frequency in ['B', '1', '2', '3', '4', '5']:
        jump = 7
    elif frequency in ['M', 'MS']:
        jump = 31
    elif frequency in ['Q']:
        jump = 92
    elif frequency in ['A']:
        jump = 366
    t0 = start_date if sample_window is None else start_date - timedelta(jump * sample_window)
    base = pd.Series([0, 0], name='timeline', index=[t0, end_date])
    if frequency in ['B', '1', '2', '3', '4', '5']:
        base = base.resample('B').last()
    else:
        base = base.resample(frequency).last()
    if frequency in ['1', '2', '3', '4', '5']:
        base = base.loc[base.index.weekday.isin([np.int(frequency) - 1])]
    if frequency == 'B' and sample_window is not None:
        n = np.sum(base.index < start_date)
        base = base.iloc[n - sample_window:]
    base = base.loc[base.index <= end_date]
    return base.fillna(0.)


def ignore_insufficient_series(data, min_size):
    '''
    Get rid of series that do not have minimum size of observations
    '''
    assert isinstance(data, pd.DataFrame)
    ans = data.loc[:, (-data.isnull()).sum(axis=0) >= min_size]
    return None if ans.empty else ans