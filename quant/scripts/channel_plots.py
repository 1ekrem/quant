from quant.lib.main_utils import *
from matplotlib import use
use("agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from quant.research import channel


def get_filename(universe):
    return os.path.expanduser('~/%s.pdf' % universe)


def expand_to_find_channel(r, lookback=42, skip_tail=2):
    k = None
    l = lookback
    while k is None:
        rs = r.iloc[-l:]
        k, b, h, f, th, tl = channel.find_channel(rs.iloc[:-skip_tail])
        if (th or tl) and l < len(r):
            l += 10
            k = None
    return rs, k, b, h


def make_channel_pdf(rtns, universe='SMX'):
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
            for lookback in [42, 252]:
                v += 1
                plt.subplot(nrow, ncol, v)
                rs, k, b, h = expand_to_find_channel(r, lookback)
                rs.cumsum().plot()
                channel.plot_channel(rs, k, b, h)
                plt.title('%s [%d]' % (c, lookback), weight='bold')
                save_fig = True
        if save_fig:
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def plot_universe(universe='SMX'):
    rtn, rs, rm, v = channel.get_dataset(universe, max_spread=None)
    make_channel_pdf(rtn, universe)


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


