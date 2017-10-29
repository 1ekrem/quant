'''
Created on 7 Oct 2017

@author: wayne
'''
from matplotlib import use
use('agg')

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as dt
from quant.scripts import stocks_mom as sm
from quant.strategies import smx
from quant.lib import visualization_utils as vu
from quant.lib.main_utils import Email


def plot_pnl(pnl17, pnl, pnl_mom):
    plt.figure(figsize=(6, 4))
    vu.axis_area_plot(pnl17.iloc[-52:].cumsum())
    plt.plot(pnl.index[-52:], pnl.iloc[-52:].cumsum(), color='black')
    plt.plot(pnl_mom.index[-52:], pnl_mom.iloc[-52:].cumsum(), color='red')
    vu.use_monthly_ticks(pnl17.iloc[-52:])
    plt.title('Cumulative PnL', weight='bold')
    plt.tight_layout()
    filename = os.path.expanduser('~/pnl.png')
    plt.savefig(filename)
    plt.close()
    return filename


def run_smx_check(capital=500):
    stock_data = sm.get_stocks_data()
    sig, sig_date = sm.run_smx_signal(stock_data, capital=capital)
    table = np.round(sig.iloc[:30], 2)
    pnl17 = sm.run_portfolio(stock_data, *sm.P17)
    pnl = sm.run_portfolio(stock_data, *sm.PL)
    pnl_mom = sm.run_momentum_portfolio(stock_data)
    table2 = np.round(100. * pd.concat([pnl17, pnl, pnl_mom], axis=1).iloc[-6:], 2)
    table2.columns = ['Short PnL (%)', 'Long PnL (%)', 'Momentum PnL (%)']
    table2.index = [x.strftime('%Y-%m-%d') for x in table2.index]
    filename = plot_pnl(pnl17, pnl, pnl_mom)
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'SMX Stocks')
    mail.add_date(dt.today())
    mail.add_image(filename, 600, 400)
    mail.add_text('PnL Summary', bold=True)
    mail.add_table(table2)
    mail.add_text('Stocks signal as of %s' % sig_date.strftime('%B %d, %Y'), bold=True)
    mail.add_table(table)
    mail.send_email()


def run_smx_ml_check(capital=500):
    sim, sig, sig_date = smx.get_smx_signal(capital=capital)
    table = np.round(sig, 2)
    pnl = sim.strategy_returns['Long Only']
    pnl_ls = sim.strategy_returns['Long Short']
    table2 = np.round(100. * pd.concat([pnl, pnl_ls], axis=1).iloc[-6:], 2)
    table2.columns = ['Long Only (%)', 'Long Short (%)']
    table2.index = [x.strftime('%Y-%m-%d') for x in table2.index]
    filename = plot_pnl(pnl, pnl_ls)
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'SMX ML')
    mail.add_date(dt.today())
    mail.add_image(filename, 600, 400)
    mail.add_text('PnL Summary', bold=True)
    mail.add_table(table2)
    mail.add_text('Stocks signal as of %s' % sig_date.strftime('%B %d, %Y'), bold=True)
    mail.add_table(table)
    mail.send_email()

def main():
    run_smx_check()
    # run_smx_ml_check()


if __name__ == '__main__':
    main()
