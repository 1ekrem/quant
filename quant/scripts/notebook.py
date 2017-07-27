'''
Created on 27 Jul 2017

@author: wayne
'''

from quant.lib import timeseries_utils as tu

DATABASE_NAME = 'quant'
STRATEGY_TABLE = 'strategies'


def get_strategy_data(strategy_name):
    returns = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Asset Returns', 'Returns'], data_name=strategy_name)
    positions = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Positions'], data_name=strategy_name)
    signal = tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Signal'], data_name=strategy_name)
    return dict(returns=returns, positions=positions, signal=signal)

