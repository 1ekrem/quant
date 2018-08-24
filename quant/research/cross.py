from quant.lib.main_utils import *
from quant.data import stocks
from quant.lib import portfolio_utils as pu, visualization_utils as vu, timeseries_utils as tu
from matplotlib import pyplot as plt
from quant.lib.timeseries_utils import fit_data


PATH = os.path.expanduser('~/TempWork/cross/')
make_dir(PATH)


def _get_first(x):
    idx = x.index[x.abs() > 0]
    if len(idx) > 0:
        ans = x.copy()
        ans[ans.index < idx[0]] = np.nan
        return ans
    else:
        return x * np.nan


def get_returns(r):
    rc = r[r.abs() < .3]
    rtn = r.resample('W').sum().apply(_get_first, axis=0)
    w = rtn.abs()
    vol = w[w > 0].rolling(52, min_periods=13).median()
    vol[vol < 5e-3] = 5e-3
    vol2 = vol.copy()
    vol2[vol2 < .02] = .02
    rv = rtn.divide(vol)
    rm = rv.subtract(rv.mean(axis=1), axis=0)
    s = r.abs().resample('W').max()
    return rtn, rm, vol2, s


def get_pos(s1, s2, vol, high, low, acc, ax, s, bottom=True, shock=False, period=True):
    if bottom:
        pos = 1. * ((s1 <= -low) & (s2 >= high) & (acc == ax))
    else:
        pos = 1. * ((s1 <= -low) & (s2 >= high))
    if shock:
        pos = pos[s <= .25]
    pos = pos.divide(vol)
    if period:
        pos = pos.fillna(0.)
        pos = pos[pos.rolling(2, min_periods=1).max().shift() == 0]
        pos = pos[pos > 0]
    return pos


def get_pnl(s1, s2, r, vol, high, low, acc, ax, s, bottom=True, shock=False, period=True):
    p = get_pos(s1, s2, vol, high, low, acc, ax, s, bottom, shock, period=period)
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
def run_long(rtn, rm, vol, sx, i, start_date, end_date, style='A', bottom=True, shock=False, period=True):
    if style == 'A':
        l = np.arange(.3, 2.01, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index]
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index]
    elif style == 'B':
        l = np.arange(.5, 3.01, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index] * np.sqrt(1. * i)
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index] * np.sqrt(52.)
    s = sx.rolling(13, min_periods=1).max()
    acc = rtn.cumsum()
    ax = acc.rolling(i, min_periods=1).min()
    ans = None
    m = -1.
    mu = -1.
    df = -1.
    for high in l:
        for low in l:
            pnl = get_pnl(s1, s2, rtn, vol, high, low, acc, ax, s, bottom, shock, period=period)
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
 

def run_long_pos(rtn, rm, vol, sx, i, start_date, end_date, style='A', bottom=True, shock=False):
    if style == 'A':
        l = np.arange(.3, 2.01, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index]
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index]
    elif style == 'B':
        l = np.arange(.5, 3.01, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index] * np.sqrt(1. * i)
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index] * np.sqrt(52.)    
    s = sx.rolling(13, min_periods=1).max()
    acc = rtn.cumsum()
    ax = acc.rolling(i, min_periods=1).min()
    ans = None
    m = -1.
    for high in l:
        for low in l:
            pnl = get_pnl(s1, s2, rtn, vol, high, low, acc, ax, s, bottom, shock)
            if pnl is not None:
                pnl = pnl[start_date:]
                tot = pnl.mean()
                if tot > m:
                    m = tot
                    ans = get_pos(s1, s2, vol, high, low, acc, ax, s, bottom, shock, period=False)[start_date:]
    return ans

    
def estimate_reversal(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), style='A', bottom=True, shock=False):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    u = stocks.get_universe(universe)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rm, vol, sx = get_returns(r)
    ana = []
    plt.figure(figsize=(10, 7))
    for i in xrange(1, 14):
        logger.info('Lookback %d' % i)
        pnl, mu, df = run_long(rtn, rm, vol, sx, i, start_date, end_date, style, bottom, shock, period=True if i > 5 else False)
        if pnl is not None:
            pnl.plot()
            ana.append([mu, df])
    plt.legend(loc='best', frameon=False)
    style_text = style + 'S' if shock else style
    plt.title('%s %s %s' % (universe, style_text, 'Bt' if bottom else 'All'), weight='bold')
    plt.tight_layout()
    plt.savefig(PATH + '%s_%s_%s_%d.png' % (universe, style_text, 'Bt' if bottom else 'All', start_date.year))
    plt.close()
    plt.figure()
    vu.bar_plot(pd.DataFrame(ana, index=np.arange(1, 14), columns=['Mean', 'Decay']).T)
    plt.plot(np.arange(len(ana)) * 3 + 1, np.sum(ana, axis=1), marker='s', color='black')
    plt.axhline(.35, color='grey', ls='--')
    plt.title('%s %s %s' % (universe, style_text, 'Bt' if bottom else 'All'), weight='bold')
    plt.tight_layout()
    plt.savefig(PATH + '%s_%s_%s_%d_decay.png' % (universe, style_text, 'Bt' if bottom else 'All', start_date.year))
    plt.close()


def run_all():
    for universe in ['SMX', 'FTSE250']:
        for style in ['A', 'B']:
            for bottom in [True, False]:
                for shock in [True, False]:
                    estimate_reversal(universe, style=style, bottom=bottom, shock=shock)


def _count_zero(x):
    i = 0
    ans = []
    for v in x:
        ans.append(i)
        if v > 0:
            i = 0
        else:
            i += 1
    return ans


def test_reversal(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), style='A', bottom=True, shock=False):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    u = stocks.get_universe(universe)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rm, vol, sx = get_returns(r)
    fr = rtn.rolling(4, min_periods=1).sum().shift(-4)
    plt.figure(figsize=(14, 7))
    for i in xrange(1, 13):
        plt.subplot(3, 4, i)
        logger.info('Lookback %d' % i)
        pos = run_long_pos(rtn, rm, vol, sx, i, start_date, end_date, style, bottom, shock)
        if pos is not None:
            p = pos.fillna(0.)
            ans = [np.mean(tu.get_observations(pos.mul(tu.fit_data(fr, pos)).fillna(0.),
                                               pos[p.rolling(j, min_periods=1).max().shift() == 0]))
                   for j in xrange(1, 14)]
            vu.bar_plot(pd.Series(ans, index=np.arange(13)+1) - 1.)
            plt.title(i)
    plt.tight_layout()
    plt.savefig(PATH + '%s_%s_%s_%d_test2.png' % (universe, style, 'Bt' if bottom else 'All', start_date.year))
    plt.close()


def test_all():
    for universe in ['FTSE250']:
        for style in ['A', 'B']:
            for bottom in [True, False]:
                test_reversal(universe, style=style, bottom=bottom)


def run_trace(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), style='A', bottom=True):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    u = stocks.get_universe(universe)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rv, rm, vol, sx = get_returns(r)
    fr = [rtn.rolling(i, min_periods=1).sum().shift(-i) for i in xrange(1, 14)]
    plt.figure(figsize=(14, 7))
    for i in xrange(1, 13):
        plt.subplot(3, 4, i)
        logger.info('Lookback %d' % i)
        pos = run_long_pos(rtn, rm, vol, i, start_date, end_date, style, bottom)
        if pos is not None:
            tmp = []
            for x in fr:
                y = tu.get_observations(pos.mul(tu.fit_data(x, pos)).fillna(0.), pos)
                tmp.append([np.mean(y)] + list(np.percentile(y, [10, 30, 70, 90])))
            tmp = pd.DataFrame(tmp, index=np.arange(len(tmp)) + 1, columns=['Average', '30%', '40%', '60%', '70%'])
            plt.fill_between(tmp.index, tmp['30%'], tmp['70%'], color='green', alpha=0.2)
            plt.fill_between(tmp.index, tmp['40%'], tmp['60%'], color='green', alpha=0.4)
            plt.plot(tmp.index, tmp.Average, lw=2, color='green')
            plt.ylim((tmp.Average.min() - .05, tmp.Average.max() + .05))
            plt.title(i, weight='bold')
    plt.tight_layout()
    plt.savefig(PATH + '%s_%s_%s_%d_trace.png' % (universe, style, 'Bt' if bottom else 'All', start_date.year))
    plt.close()


def run_all_traces():
    for universe in ['SMX', 'FTSE250']:
        for style in ['A', 'B']:
            for bottom in [True, False]:
                run_trace(universe, style=style, bottom=bottom)