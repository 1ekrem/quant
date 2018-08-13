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


def get_pos(s1, s2, vol, high, low, acc, ax, sig2=True):
    if sig2:
        pos = 1. * ((s1 <= -low) & (s2 >= high) & (acc == ax))
    else:
        pos = 1. * ((s1 <= -low) & (s2 >= high))
    return pos.divide(vol)


def get_pnl(s1, s2, r, vol, high, low, acc, ax, sig2=True):
    p = get_pos(s1, s2, vol, high, low, acc, ax, sig2)
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
def run_long(r, rm, vol, i, start_date, end_date, sig2=True):
    l = np.arange(.3, 1.51, .1)
    s1 = rm.rolling(i).mean().loc[r.index]
    s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[r.index]
    acc = r.cumsum()
    ax = acc.rolling(i, min_periods=1).min()
    ans = None
    m = -1.
    for high in l:
        for low in l:
            pnl = get_pnl(s1, s2, r, vol, high, low, acc, ax, sig2)
            if pnl is not None:
                pnl = pnl[start_date:]
                tot = pnl.mean()
                df = (pnl[end_date:].mean() - pnl[:end_date].mean()) * 52.
                if tot > m:
                    m = tot
                    mu = pnl.mean() * 52.
                    sr = pnl.mean() / pnl.std() * np.sqrt(52.)
                    pnl.name = '%d %.2f %.1f%% %.1f%% [%.1f %.1f]' % (i, sr, 100. * mu, 100. * df, high, -low)
                    ans = pnl.cumsum()
    return ans


def run_long_pos(r, rm, vol, i, start_date, end_date, sig2=True):
    l = np.arange(.3, 1.51, .1)
    s1 = rm.rolling(i).mean().loc[r.index]
    s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[r.index]
    acc = r.cumsum()
    ax = acc.rolling(i, min_periods=1).min()
    ans = None
    m = -1.
    for high in l:
        for low in l:
            pnl = get_pnl(s1, s2, r, vol, high, low, acc, ax, sig2)
            if pnl is not None:
                tot = pnl[start_date:].mean()
                if tot > m:
                    m = tot
                    ans = get_pos(s1, s2, vol, high, low, acc, ax, sig2)[start_date:]
    return ans

    
def estimate_reversal(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), sig2=True):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    u = stocks.get_universe(universe)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rv, rm, vol = get_returns(r)
    for i in xrange(1, 10):
        logger.info('Lookback %d' % i)
        pnl = run_long(rtn, rm, vol, i, start_date, end_date, sig2=sig2)
        if pnl is not None:
            pnl.plot()
    plt.legend(loc='best', frameon=False)


def test_reversal_speed(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), sig2=True):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    u = stocks.get_universe(universe)
    r = r.loc[:, r.columns.isin(u.index)]
    rtn, rv, rm, vol = get_returns(r)
    fr = rtn.rolling(4, min_periods=1).sum().shift(-4)
    plt.figure(figsize=(12, 7))
    for i in xrange(1, 10):
        s = rm.rolling(26, min_periods=13).mean().shift(i)
        plt.subplot(3, 3, i)
        logger.info('Lookback %d' % i)
        pos = run_long_pos(rtn, rm, vol, i, start_date, end_date, sig2=sig2)
        if pos is not None:
            x = tu.get_observations(s.fillna(0.), pos)
            y = tu.get_observations(pos.mul(tu.fit_data(fr, pos)).fillna(0.), pos)
            vu.binned_statistic_plot(x, y, 'mean', 17, (-3, 3))
            plt.xlim((-7.5, 17.5))
            plt.title(i)
    plt.tight_layout()