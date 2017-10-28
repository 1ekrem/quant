'''
Created on 26 Jul 2017

@author: wayne
'''
import sys
import numpy as np
import pandas as pd
from datetime import datetime as dt
from quant.lib import timeseries_utils as tu, machine_learning_utils as mu, portfolio_utils as pu
from quant.research import machine_learning as ml
from quant.lib.main_utils import MODEL_PATH

DATABASE_NAME = 'quant'
STRATEGY_TABLE = 'strategies'
START_DATE = dt(2007, 1, 1)
SAMLE_DATE = dt(2014, 1, 1)
FORECAST_HORIZON = 4

def estimate_smx_model(load_model=False):
    universe='SMX Index'
    simulation_name = 'SMX'
    strategy_component = mu.StockRandomBoostingComponent
    sim = ml.StocksSim(start_date=START_DATE, end_date=dt.today(), sample_date=SAMLE_DATE,
                       forecast_horizon=FORECAST_HORIZON, strategy_component=strategy_component,
                       universe=universe, simulation_name=simulation_name, max_depth=13,
                       model_path=MODEL_PATH, load_model=load_model, cross_validation=True,
                       cross_validation_buskcets=10, smart_cross_validation=True)
    return sim


def create_smx_model():
    sim = estimate_smx_model()
    sim.pickle_model()


def get_smx_signal(capital=500):
    sim = estimate_smx_model(True)
    s = sim.signals.iloc[-1].dropna().sort_values(ascending=False)
    s = s.iloc[:30].to_frame()
    s.columns = ['Signal']
    for idx in s.index:
        r = sim.stock_returns.get(idx)
        s.loc[idx, 'Multiplier'] = capital / r.Vol.iloc[-1]
    return sim, s, sim.signals.index[-1]


def main():
    pass

if __name__ == '__main__':
    main()

    