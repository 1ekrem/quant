'''
Created on 1 Oct 2017

@author: wayne
'''
import os
import pandas as pd
from matplotlib import use
use('agg')
from datetime import datetime as dt
from matplotlib import pyplot as plt
from quant.scripts import notebook
from quant.lib.main_utils import Email


def check_strategy(strategy='SPX_FUTURE'):
    data = notebook.get_trading_strategy_data(strategy)
    table = pd.concat([data['positions'], data['returns']], axis=1).iloc[-5:]
    table.index = [x.strftime('%Y-%m-%d') for x in table.index]
    notebook.plot_short_returns(data, 63, True, 'Original Returns', 'Reversal Returns')
    filename = os.path.expanduser('~/spx_future_returns.png')
    plt.savefig(filename)
    plt.close()
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'SPX Futures')
    mail.add_date(dt.today())
    mail.add_image(filename, 660, 440)
    mail.add_table(table)
    mail.send_email()
    os.remove(filename)


def main():
    check_strategy()


if __name__ == '__main__':
    main()
