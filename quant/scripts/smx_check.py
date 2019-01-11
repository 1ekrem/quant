'''
Created on 7 Oct 2017

@author: wayne
'''
from matplotlib import use
use('agg')

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as dt
from quant.scripts import stocks_mom as sm
from quant.strategies import smx
from quant.lib import timeseries_utils as tu, visualization_utils as vu
from quant.lib.main_utils import Email


def plot_pnl(pnls):
    plt.figure(figsize=(6, 4))
    pnl = pd.concat(pnls, axis=1)
    vu.axis_area_plot(pnl.iloc[-52:, 0].cumsum())
    for i in xrange(1, len(pnl.columns)):
        plt.plot(pnl.index[-52:], pnl.iloc[-52:, i].cumsum(), label=pnl.columns[i])
    vu.use_monthly_ticks(pnl.iloc[-52:])
    plt.legend(loc='best', frameon=False)
    plt.title('Cumulative PnL', weight='bold')
    plt.tight_layout()
    filename = os.path.expanduser('~/pnl.png')
    plt.savefig(filename)
    plt.close()
    return filename


def run_smx_check(capital=200):
    r, rm, posvol, volume = sm.get_smx_data()
    f = sm.get_fundamentals('SMX').reindex(r.columns, axis=1)
    score = tu.resample(f, r)
    sig_date, pos, pnls = sm.run_package(r, rm, posvol, score, capital)
    table = np.round(100. * pd.concat(pnls, axis=1).iloc[-6:], 2)
    table.columns = [x + ' (%)' for x in table.columns]
    table.index = [x.strftime('%Y-%m-%d') for x in table.index]
    filename = plot_pnl(pnls)
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'SMX Stocks')
    mail.add_date(dt.today())
    mail.add_image(filename, 600, 400)
    mail.add_text('PnL Summary', bold=True)
    mail.add_table(table, width=700)
    for i, x in enumerate(pos):
        mail.add_text('%s Positions' % pnls[i].name)
        table2 = np.round(x.fillna(0.))
        mail.add_table(table2, width=400)
    mail.send_email()


def run_smx_check2(capital=200):
    r, rm, posvol, volume = sm.get_smx_data()
    f = sm.get_fundamentals('SMX').reindex(r.columns, axis=1)
    score = tu.resample(f, r)
    sig_date, pos, pnls = sm.run_package2(r, rm, posvol, score, capital)
    table = np.round(100. * pd.concat(pnls, axis=1).iloc[-6:], 2)
    table.columns = [x + ' (%)' for x in table.columns]
    table.index = [x.strftime('%Y-%m-%d') for x in table.index]
    filename = plot_pnl(pnls)
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'SMX Stocks 2')
    mail.add_date(dt.today())
    mail.add_image(filename, 600, 400)
    mail.add_text('PnL Summary', bold=True)
    mail.add_table(table, width=700)
    for i, x in enumerate(pos):
        mail.add_text('%s Positions' % pnls[i].name)
        table2 = np.round(x.fillna(0.))
        mail.add_table(table2, width=400)
    mail.send_email()


def run_ftse250_check(capital=200):
    r, rm, posvol, volume = sm.get_ftse250_data()
    f = sm.get_fundamentals('FTSE250')
    score = tu.resample(f, r)
    sig_date, pos, pnls = sm.run_package(r, rm, posvol, volume, score, capital)
    table = np.round(100. * pd.concat(pnls, axis=1).iloc[-6:], 2)
    table.columns = [x + ' (%)' for x in table.columns]
    table.index = [x.strftime('%Y-%m-%d') for x in table.index]
    filename = plot_pnl(pnls)
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'FTSE250 Stocks')
    mail.add_date(dt.today())
    mail.add_image(filename, 600, 400)
    mail.add_text('PnL Summary', bold=True)
    mail.add_table(table, width=700)
    for i, x in enumerate(pos):
        mail.add_text('%s Positions' % pnls[i].name)
        table2 = np.round(x.fillna(0.))
        mail.add_table(table2, width=400)
    mail.send_email()


def run_aim_check(capital=200):
    r, rm, posvol, volume = sm.get_aim_data()
    sig_date, pos, pnls = sm.run_package(r, rm, posvol, volume, capital)
    table = np.round(100. * pd.concat(pnls, axis=1).iloc[-6:], 2)
    table.columns = [x + ' (%)' for x in table.columns]
    table.index = [x.strftime('%Y-%m-%d') for x in table.index]
    filename = plot_pnl(pnls)
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'AIM Stocks')
    mail.add_date(dt.today())
    mail.add_image(filename, 600, 400)
    mail.add_text('PnL Summary', bold=True)
    mail.add_table(table, width=700)
    for i, x in enumerate(pos):
        mail.add_text('%s Positions' % pnls[i].name)
        table2 = np.round(x.fillna(0.))
        mail.add_table(table2, width=400)
    mail.send_email()


def main():
    target = 'SMX'
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if target == 'SMX':
        run_smx_check()
    elif target == 'FTSE250':
        run_ftse250_check()
    elif target == 'AIM':
        run_aim_check()


if __name__ == '__main__':
    main()
