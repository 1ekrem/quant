from quant.lib.main_utils import *
from quant.data import stocks
from quant.lib import portfolio_utils as pu, visualization_utils as vu, timeseries_utils as tu
from matplotlib import pyplot as plt


def _get_first(x):
    idx = x.index[x.abs() > 0]
    if len(idx) > 0:
        ans = x.copy()
        ans[ans.index < idx[0]] = np.nan
        return ans
    else:
        return x * np.nan


def get_returns(r):
    rtn = r.resample('W').sum().apply(_get_first, axis=0)
    w = rtn.abs()
    vol = w[w > 0].rolling(52, min_periods=13).median()
    vol[vol < 5e-3] = 5e-3
    vol2 = vol.copy()
    vol2[vol2 < .02] = .02
    rv = rtn.divide(vol)
    rm = rv.subtract(rv.mean(axis=1), axis=0)
    return rtn, rv, rm, vol2


def get_pos(s1, s2, vol, high, low, acc, ax, bottom=True):
    if bottom:
        pos = 1. * ((s1 <= -low) & (s2 >= high) & (acc == ax))
    else:
        pos = 1. * ((s1 <= -low) & (s2 >= high))
    return pos.divide(vol)


def get_pnl(s1, s2, r, vol, high, low, acc, ax, bottom=True):
    p = get_pos(s1, s2, vol, high, low, acc, ax, bottom)
    if p.abs().sum(axis=1).sum() > 0:
        p = p[p.abs() > 0].ffill(limit=3)
        g = p.abs().sum(axis=1)
        pnl = r.mul(p.shift()).sum(axis=1)
        pnl /= g.shift()
        pnl = _get_first(pnl).fillna(0.)
        return pnl
    else:
        return None


# Lookback, timing bottom
def run_long(r, rm, vol, i, start_date, end_date, style='A', bottom=True):
    if style == 'A':
        l = np.arange(.3, 1.51, .1)
        s1 = rm.rolling(i).mean().loc[r.index]
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[r.index]
    elif style == 'B':
        l = np.arange(.5, 2.01, .1)
        s1 = rm.rolling(i, min_periods=1).sum().loc[r.index] / np.sqrt(1. * i)
        s2 = rm.rolling(52, min_periods=13).sum().shift(i).loc[r.index]  / np.sqrt(52.)
    acc = r.cumsum()
    ax = acc.rolling(i, min_periods=1).min()
    ans = None
    m = -1.
    mu = -1.
    df = -1.
    for high in l:
        for low in l:
            pnl = get_pnl(s1, s2, r, vol, high, low, acc, ax, bottom)
            if pnl is not None:
                pnl = pnl[start_date:]
                tot = pnl.mean()
                if tot > m:
                    m = tot
                    mu = pnl.mean() * 52.
                    df = (pnl[end_date:].mean() - pnl[:end_date].mean()) * 52.
                    sr = pnl.mean() / pnl.std() * np.sqrt(52.)
                    pnl.name = '%d %.2f %.1f%% %.1f%% [%.1f %.1f]' % (i, sr, 100. * mu, 100. * df, high, -low)
                    ans = pnl.cumsum()
    return ans, mu, df


def run_long_pos(r, rm, vol, i, start_date, end_date, style='A', bottom=True):
    if style == 'A':
        l = np.arange(.3, 1.51, .1)
        s1 = rm.rolling(i).mean().loc[r.index]
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[r.index]
    elif style == 'B':
        l = np.arange(.5, 2.01, .1)
        s1 = rm.rolling(i, min_periods=1).sum().loc[r.index] / np.sqrt(1. * i)
        s2 = rm.rolling(52, min_periods=13).sum().shift(i).loc[r.index]  / np.sqrt(52.)    
    acc = r.cumsum()
    ax = acc.rolling(i, min_periods=1).min()
    ans = None
    m = -1.
    for high in l:
        for low in l:
            pnl = get_pnl(s1, s2, r, vol, high, low, acc, ax, bottom)
            if pnl is not None:
                tot = pnl[start_date:].mean()
                if tot > m:
                    m = tot
                    ans = get_pos(s1, s2, vol, high, low, acc, ax, bottom)[start_date:]
    return ans

    
def estimate_reversal(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), style='A', bottom=True):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    u = stocks.get_universe(universe)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rv, rm, vol = get_returns(r)
    ana = []
    plt.figure(figsize=(10, 7))
    for i in xrange(1, 14):
        logger.info('Lookback %d' % i)
        pnl, mu, df = run_long(rtn, rm, vol, i, start_date, end_date, style, bottom)
        if pnl is not None:
            pnl.plot()
            ana.append([mu, df])
    plt.legend(loc='best', frameon=False)
    plt.title('%s %s %s' % (universe, style, 'Bt' if bottom else 'All'), weight='bold')
    plt.tight_layout()
    plt.savefig('%s_%s_%s_%d.png' % (universe, style, 'Bt' if bottom else 'All', start_date.year))
    plt.close()
    plt.figure()
    vu.bar_plot(pd.DataFrame(ana, index=np.arange(1, 14), columns=['Mean', 'Decay']).T)
    plt.plot(np.arange(len(ana)) * 3 + 1, np.sum(ana, axis=1), marker='s', color='black')
    plt.axhline(.35, color='grey', ls='--')
    plt.title('%s %s %s' % (universe, style, 'Bt' if bottom else 'All'), weight='bold')
    plt.tight_layout()
    plt.savefig('%s_%s_%s_%d_decay.png' % (universe, style, 'Bt' if bottom else 'All', start_date.year))
    plt.close()


def run_all():
    for universe in ['SMX', 'FTSE250']:
        for style in ['A', 'B']:
            for bottom in [True, False]:
                estimate_reversal(universe, style=style, bottom=bottom)


def test_reversal(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), style='A', bottom=True):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    u = stocks.get_universe(universe)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rv, rm, vol = get_returns(r)
    fr = rtn.rolling(4, min_periods=1).sum().shift(-4)
    plt.figure(figsize=(14, 7))
    for i in xrange(1, 13):
        s = rtn.rolling(13, min_periods=1).sum().shift(i) / np.sqrt(13.)
        plt.subplot(3, 4, i)
        logger.info('Lookback %d' % i)
        pos = run_long_pos(rtn, rm, vol, i, start_date, end_date, style, bottom)
        if pos is not None:
            x = tu.get_observations(s.loc[pos.index].fillna(0.), pos)
            y = tu.get_observations(pos.mul(tu.fit_data(fr, pos)).fillna(0.), pos)
            vu.binned_statistic_plot(x, y, 'mean', 9, (-1, 1))
            #plt.xlim((-7.5, 17.5))
            plt.title(i)
    plt.tight_layout()
    plt.savefig('%s_%s_%s_%d_test.png' % (universe, style, 'Bt' if bottom else 'All', start_date.year))
    plt.close()
