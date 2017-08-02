'''
Created on 26 Jul 2017

@author: wayne
'''
import numpy as np
from datetime import datetime as dt
from quant.data import fred, quandldata
from quant.lib import timeseries_utils as tu, machine_learning_utils as mu, portfolio_utils as pu
from quant.research import machine_learning as ml
from quant.lib.main_utils import MODEL_PATH

DATABASE_NAME = 'quant'
STRATEGY_TABLE = 'strategies'
START_DATE = dt(2000, 1, 1)
SAMLE_DATE = dt(2017, 1, 1)
DATA_FREQUENCY = '2'
FORECAST_HORIZON = 1


def spx_data_loader(*args, **kwargs):
    ans = tu.get_timeseries(DATABASE_NAME, quandldata.QUANDL_FUTURES, column_list=['settle'], data_name='S&P 500')
    ans.columns = ['SPX Index']
    return ans


def estimate_model(load_model=False):
    simulation_name = 'TEST'
    econ = fred.get_fred_us_econ_list()
    input_data_loader = fred.fred_combined_loader
    strategy_component = mu.RandomBoostingComponent
    position_component = pu.SimpleLongShort
    simple_returns=True
    cross_validation=True
    cross_validation_params=[{}] + [{'span': x} for x in np.arange(1, 14)]
    cross_validation_buckets=10
    sim = ml.EconSim(start_date=START_DATE, end_date=dt.today(), sample_date=SAMLE_DATE, data_frequency=DATA_FREQUENCY,
                     forecast_horizon=2, assets=['SPX Index'], asset_data_loader=spx_data_loader,
                     inputs=econ, input_data_loader=input_data_loader, strategy_component=strategy_component,
                     position_component=position_component, simulation_name=simulation_name, model_path=MODEL_PATH,
                     load_model=load_model, simple_returns=simple_returns, cross_validation=cross_validation,
                     cross_validation_params=cross_validation_params, cross_validation_buckets=cross_validation_buckets)
    return sim


def create_model():
    sim = estimate_model()
    sim.pickle_model()


def export_model_data(sim):
    data_name = sim.simulation_name
    data = sim.signal.copy()
    data.columns = ['Signal']
    tu.store_timeseries(data, DATABASE_NAME, STRATEGY_TABLE, data_name)
    data = sim.positions.copy()
    data.columns = ['Positions']
    tu.store_timeseries(data, DATABASE_NAME, STRATEGY_TABLE, data_name)
    data = sim.strategy_returns.copy()
    data.columns = ['Returns']
    tu.store_timeseries(data, DATABASE_NAME, STRATEGY_TABLE, data_name)
    data = sim.asset_returns.copy()
    data.columns = ['Asset Returns']
    tu.store_timeseries(data, DATABASE_NAME, STRATEGY_TABLE, data_name)


def update_pension_model():
    sim = estimate_model(True)
    export_model_data(sim)


def main():
    update_pension_model()


if __name__ == '__main__':
    main()

    