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
from quant.lib import visualization_utils as vu
from quant.lib.main_utils import Email


def plot_pnl(pnls):
    plt.figure(figsize=(6, 4))
    pnl = pnls[0]
    vu.axis_area_plot(pnl.iloc[-52:].cumsum())
    if len(pnl) > 1:
        for pnl2 in pnls[1:]:
            plt.plot(pnl2.index[-52:], pnl2.iloc[-52:].cumsum(), label=pnl2.name)
    vu.use_monthly_ticks(pnl.iloc[-52:])
    plt.legend(loc='best', frameon=False)
    plt.title('Cumulative PnL', weight='bold')
    plt.tight_layout()
    filename = os.path.expanduser('~/pnl.png')
    plt.savefig(filename)
    plt.close()
    return filename


def run_smx_check(capital=200):
    r, v, v2, s = sm.get_smx_data()
    pos, pos2, pos3, pos4, pos5, pos6, sig_date, pnl, pnl2, pnl3, pnl4, pnl5, pnl6 = sm.run_new_smx(r, v, v2, s, capital=capital)
    pnl_idx = capital * r.mean(axis=1)
    pnl_idx.name = 'Index'
    table = np.round(100. * pd.concat([pnl, pnl2, pnl3, pnl4, pnl5, pnl6, pnl_idx], axis=1).iloc[-6:], 2)
    table.columns = ['A2 (%)', 'A6 (%)', 'A7 (%)', 'B2 (%)', 'B6 (%)', 'B7 (%)', 'Index (%)']
    table.index = [x.strftime('%Y-%m-%d') for x in table.index]
    table2 = np.round(pos.fillna(0.))
    table3 = np.round(pos2.fillna(0.))
    table4 = np.round(pos3.fillna(0.))
    table5 = np.round(pos4.fillna(0.))
    table6 = np.round(pos5.fillna(0.))
    table7 = np.round(pos6.fillna(0.))
    filename = plot_pnl([pnl, pnl2, pnl3, pnl4, pnl5, pnl6, pnl_idx])
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'SMX Stocks')
    mail.add_date(dt.today())
    mail.add_image(filename, 600, 400)
    mail.add_text('PnL Summary', bold=True)
    mail.add_table(table, width=700)
    mail.add_text('A2 Positions')
    mail.add_table(table2, width=400)
    mail.add_text('A6 Positions')
    mail.add_table(table3, width=400)
    mail.add_text('A7 Positions')
    mail.add_table(table4, width=400)
    mail.add_text('B2 Positions')
    mail.add_table(table5, width=400)
    mail.add_text('B6 Positions')
    mail.add_table(table6, width=400)
    mail.add_text('B7 Positions')
    mail.add_table(table7, width=400)
    mail.send_email()


def run_ftse250_check(capital=200):
    r, v, v2, s = sm.get_ftse250_data()
    pos, pos2, pos3, pos4, sig_date, pnl, pnl2, pnl3, pnl4 = sm.run_ftse250(r, v, v2, s, capital=capital)
    pnl_idx = capital * r.mean(axis=1)
    pnl_idx.name = 'Index'
    table = np.round(100. * pd.concat([pnl, pnl2, pnl3, pnl4, pnl_idx], axis=1).iloc[-6:], 2)
    table.columns = ['A4 (%)', 'A6 (%)', 'B4 (%)', 'B6 (%)', 'Index (%)']
    table.index = [x.strftime('%Y-%m-%d') for x in table.index]
    table2 = np.round(pos.fillna(0.))
    table3 = np.round(pos2.fillna(0.))
    table4 = np.round(pos3.fillna(0.))
    table5 = np.round(pos4.fillna(0.))
    filename = plot_pnl([pnl, pnl2, pnl3, pnl4, pnl_idx])
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'FTSE250 Stocks')
    mail.add_date(dt.today())
    mail.add_image(filename, 600, 400)
    mail.add_text('PnL Summary', bold=True)
    mail.add_table(table, width=700)
    mail.add_text('A4 Positions')
    mail.add_table(table2, width=400)
    mail.add_text('A6 Positions')
    mail.add_table(table3, width=400)
    mail.add_text('B4 Positions')
    mail.add_table(table4, width=400)
    mail.add_text('B6 Positions')
    mail.add_table(table5, width=400)
    mail.send_email()


def main():
    target = 'SMX'
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if target == 'SMX':
        run_smx_check()
    elif target == 'FTSE250':
        run_ftse250_check()


if __name__ == '__main__':
    main()
