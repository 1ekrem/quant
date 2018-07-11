import matplotlib
matplotlib.use('Agg')

from quant.lib.main_utils import *
from quant.research import equity
from quant.lib import timeseries_utils as tu


def load_smx_model():
    end_date = dt.today()
    start_date = end_date - relativedelta(years=3)
    sim = equity.MomentumSim(start_date, end_date, start_date, 'SMX', 'SMX', load_model=True)
    return sim


def plot_pnl(m):
    filename = os.path.expanduser('~/pnl.png')
    mu = m.stock_returns.mean(axis=1)
    mu *= tu.resample((1. / m.stock_vol).mean(axis=1), mu)
    acc = pd.concat([m.pnl, m.market_neutral_pnl, mu], axis=1)
    acc.columns = ['PnL', 'Market Neutral', 'Index']
    acc[m.start_date:].cumsum().plot()
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename


def get_output(m):
    filename = os.path.expanduser('~/signal.csv')
    s = m.signals.iloc[-1]
    p = 1. / m.stock_vol.iloc[-1]
    tmp = pd.concat([s, p], axis=1)
    tmp.columns = ['Signal', 'Position']
    tmp.dropna().sort_values('Signal').to_csv(filename)
    return filename

    
def run_smx_check():
    m = load_smx_model()
    filename = plot_pnl(m)
    fname = get_output(m)
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'SMX ML')
    mail.add_date(dt.today())
    mail.add_image(filename, 600, 400)
    table3 = np.round(pd.concat([m.pnl, m.market_neutral_pnl], axis=1).resample('W').sum(), 1)
    table3.columns = ['PnL', 'Market Neutral PnL']
    table = m._pos.iloc[-1].dropna().sort_values().to_frame()
    table.index.name = 'Signal'
    table2 = np.round(m.positions.iloc[-1].dropna().sort_values().to_frame(), 1)
    table.index.name = 'Positions'
    mail.add_text('PnL')
    mail.add_table(table3, width=600)
    mail.add_text('Signal Positions')
    mail.add_table(table, width=400)
    mail.add_text('Current Positions')
    mail.add_table(table2, width=400)
    mail.add_attachment(fname)
    mail.send_email()
    os.remove(filename)
    os.remove(fname)


def main():
    run_smx_check()


if __name__ == '__main__':
    main()
