'''
Created on 27 Sep 2017

@author: wayne
'''
from quant.lib import timeseries_utils as tu
from quant.data import stocks


def get_universe(universe):
    return tu.get_description(stocks.DATABASE_NAME, stocks.STOCKS_DESCRIPTION, [universe])


def load_stock_returns(universe):
    return dict([(idx, tu.get_timeseries(stocks.DATABASE_NAME, stocks.STOCK_RETURNS, data_name=idx)) for idx in universe.index])


def run_smx():
    universe = get_universe('SMX Index')
    stock_data = load_stock_returns(universe)
    return stock_data