'''
Created on 26 Jul 2017

@author: wayne
'''
from datetime import datetime as dt
from quant.data import quandldata
from quant.lib import timeseries_utils as tu, portfolio_utils as pu


DATABASE_NAME = 'quant'
STRATEGY_TABLE = 'strategies'
START_DATE = dt(2000, 1, 1)
SAMLE_DATE = dt(2017, 1, 1)
DATA_FREQUENCY = 'B'


def usd_data_loader(*args, **kwargs):
    ans = tu.get_timeseries(DATABASE_NAME, quandldata.QUANDL_FUTURES, column_list=['settle'], data_name='US Dollar')
    ans.columns = ['US Dollar']
    return ans


def usd_signal_loader(*args, **kwargs):
    return tu.get_timeseries(DATABASE_NAME, STRATEGY_TABLE, column_list=['Signal'], data_name='USD')


def estimate_model():
    simulation_name = 'USD_FUTURE'
    signal_loader = usd_signal_loader
    sim = pu.TradingSim(start_date=START_DATE, end_date=dt.today(), data_frequency=DATA_FREQUENCY,
                        assets=['US Dollar'], asset_data_loader=usd_data_loader,
                        signal_loader=signal_loader, simulation_name=simulation_name)
    return sim


def export_model_data(sim):
    data_name = sim.simulation_name
    data = sim.signal.copy()
    data.columns = ['Signal']
    tu.store_timeseries(data, DATABASE_NAME, STRATEGY_TABLE, data_name)
    data = sim.positions.copy()
    data.columns = [x + ' Positions' for x in data.columns]
    tu.store_timeseries(data, DATABASE_NAME, STRATEGY_TABLE, data_name)
    data = sim.strategy_returns.copy()
    data.columns = [x + ' Returns' for x in data.columns]
    tu.store_timeseries(data, DATABASE_NAME, STRATEGY_TABLE, data_name)
    data = sim.asset_returns.copy()
    data.columns = ['Asset Returns']
    tu.store_timeseries(data, DATABASE_NAME, STRATEGY_TABLE, data_name)


def update_model():
    sim = estimate_model()
    export_model_data(sim)


def main():
    update_model()


if __name__ == '__main__':
    main()

    