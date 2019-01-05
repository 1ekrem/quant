from quant.lib.main_utils import *
from matplotlib import use
use("agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from quant.research import channel
from quant.data import stocks


def get_filename(universe):
    return os.path.expanduser('~/%s.pdf' % universe)


def get_two_plots(r):
    if r.empty:
        return []
    else:
        channel_set = channel.get_channel_set(r)
        run_set = []
        backup_set = []
        for i in xrange(len(channel_set)):
            p = channel_set[i][3].get('points')
            rs = channel_set[i][4]
            if p > 3:
                run_set.append((len(rs), i))
            else:
                backup_set.append((len(rs), i))
        if len(run_set) < 2:
            x = np.min((2 - len(run_set), len(backup_set)))
            run_set = run_set + backup_set[-x:]
        run_set = sorted(run_set)
        a = channel_set[run_set[0][1]] if len(run_set) > 0 else None
        b = channel_set[run_set[1][1]] if len(run_set) > 1 else None
        return (a, b)
 
 
def make_channel_pdf(rtns, universe='SMX', t4=None, t52=None):
    filename = get_filename(universe)
    ncol = 2
    nrow = 4
    i = 0
    save_fig = False
    with PdfPages(filename) as pdf:
        for j, c in enumerate(rtns.columns):
            logger.info('Charting %s' % c)
            r = rtns.loc[:, c].dropna()
            k = j % nrow
            if k == 0:
                if save_fig:
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                plt.figure(figsize=(ncol * 5., nrow * 3.5))
                v = 0
            for x in get_two_plots(r):
                v += 1
                plt.subplot(nrow, ncol, v)
                if x is not None:
                    k, b, h, ans, rs = x
                    rs.cumsum().plot(label='%d %d' % (ans.get('points'), ans.get('signal')))
                    channel.plot_channel(rs, k, b, h)
                    plt.legend(loc='best', frameon=False)
                    tscript = ' '
                    if t4 is not None:
                        if ~np.isnan(t4.loc[c]):
                            tscript += '%.2f ' % t4.loc[c]
                    if t52 is not None:
                        if ~np.isnan(t52.loc[c]):
                            tscript += '%.2f ' % t52.loc[c]
                    plt.title('%s%s[%dM]' % (c, tscript, len(rs) / 21), weight='bold')
                save_fig = True
        if save_fig:
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def plot_universe(universe='SMX'):
    rtn, rs, rm, v = channel.get_dataset(universe, max_spread=None)
    t4 = stocks.load_google_returns(data_table=stocks.UK_ESTIMATES, data_name='T4')
    t52 = stocks.load_google_returns(data_table=stocks.UK_ESTIMATES, data_name='T52')
    t4 = t4.reindex(rtn.columns, axis=1)
    t52 = t52.reindex(rtn.columns, axis=1)
    make_channel_pdf(rtn[dt(2010,1,1):], universe, t4.iloc[-1], t52.iloc[-1])


def send_channel_email():
    plot_universe('SMX')
    plot_universe('FTSE250')
    mail = Email('wayne.cq@hotmail.com', ['wayne.cq@hotmail.com'], 'Channels')
    mail.add_date(dt.today())
    mail.add_text('SMX and FTSE250')
    mail.add_attachment(get_filename('SMX'))
    mail.add_attachment(get_filename('FTSE250'))
    mail.send_email()
    os.remove(get_filename('SMX'))
    os.remove(get_filename('FTSE250'))


def main():
    send_channel_email()


if __name__ == '__main__':
    main()


