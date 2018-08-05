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


def plot_pnl(pnl, pnl2, pnl_idx):
    plt.figure(figsize=(6, 4))
    vu.axis_area_plot(pnl.iloc[-52:].cumsum())
    plt.plot(pnl2.index[-52:], pnl.iloc[-52:].cumsum(), color='black', label='Rev')
    plt.plot(pnl_idx.index[-52:], pnl_idx.iloc[-52:].cumsum(), color='green', label='Index')
    vu.use_monthly_ticks(pnl.iloc[-52:])
    plt.legend(loc='best', frameon=False)
    plt.title('Cumulative PnL', weight='bold')
    plt.tight_layout()
    filename = os.path.expanduser('~/pnl.png')
    plt.savefig(filename)
    plt.close()
    return filename


def run_smx_check(capital=500 * 1.9):
    r, v, v2 = sm.get_smx_data()
    sig, pos, pos2, sig_date, pnl, pnl2 = sm.run_new_smx(r, v, v2, capital=capital)
    fname = os.path.expanduser('~/signal.csv')
    sig.to_csv(fname)
    pnl_idx = capital * r.mean(axis=1)
    table = np.round(100. * pd.concat([pnl, pnl2, pnl_idx], axis=1).iloc[-6:], 2)
    table.columns = ['3 Week PnL (%)', '4 Week PnL (%)', 'Index PnL (%)']
    table.index = [x.strftime('%Y-%m-%d') for x in table.index]
    table2 = np.round(pos[pos > 0].to_frame())
    table3 = np.round(pos2[pos2 > 0].to_frame())
    filename = plot_pnl(pnl, pnl2, pnl_idx)
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'SMX Stocks')
    mail.add_date(dt.today())
    mail.add_image(filename, 600, 400)
    mail.add_text('PnL Summary', bold=True)
    mail.add_table(table, width=600)
    mail.add_text('3 Week Positions')
    mail.add_table(table2, width=400)
    mail.add_text('4 Week Positions')
    mail.add_table(table3, width=400)
    mail.add_attachment(fname)
    mail.send_email()


def main():
    run_smx_check()


if __name__ == '__main__':
    main()
