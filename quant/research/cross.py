from quant.lib.main_utils import *
from quant.data import stocks
from quant.lib import portfolio_utils as pu, visualization_utils as vu, timeseries_utils as tu
from matplotlib import pyplot as plt
from quant.lib.timeseries_utils import fit_data
from numpy.distutils.log import good
from numpy.ma.core import get_data


PATH = os.path.expanduser('~/TempWork/cross/')
make_dir(PATH)


def get_dataset(universe):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    if universe == 'SMX':
        u = stocks.get_ftse_smx_universe()
    elif universe == 'FTSE250':
        u = stocks.get_ftse250_universe()
    r = r.loc[:, r.columns.isin(u.index)]
    return get_returns(r)


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
    sx = 1. * (s.rolling(13, min_periods=1).max() <= .25)
    return rtn, rm, vol2, sx


def get_pos(s1, s2, vol, high, low, acc, ax, s, x, bottom=True, shock=False, enhance=True):
    if bottom:
        pos = 1. * ((s1 <= -low) & (s2 >= high) & (acc == ax))
    else:
        pos = 1. * ((s1 <= -low) & (s2 >= high))
    if shock:
        pos = pos[s > 0]
    if enhance:
        pos = pos[x > 0]
    pos = pos.divide(vol)
    return pos


def get_pnl(s1, s2, r, vol, high, low, acc, ax, s, x, bottom=True, shock=False, enhance=True, holding_period=3):
    p = get_pos(s1, s2, vol, high, low, acc, ax, s, x, bottom, shock, enhance)
    if p.abs().sum(axis=1).sum() > 0:
        p = p[p.abs() > 0].ffill(limit=holding_period)
        g = p.abs().sum(axis=1)
        pnl = r.mul(p.shift()).sum(axis=1)
        pnl /= g.shift()
        pnl = _get_first(pnl).fillna(0.)
        return pnl
    else:
        return None


# blind factor identification
def get_svd_loadings(rm):
    r = rm.subtract(rm.mean(axis=1), axis=0).fillna(0.).T
    u, s, v = np.linalg.svd(r)
    return pd.DataFrame(u, index=rm.columns).iloc[:, :len(s)]


def get_stock_stm(rm):
    m = rm.rolling(8).sum()
    return np.sign(m.subtract(m.mean(axis=1), axis=0))


def get_stmom_weight(u, rm, sm, lookback=1):
    ans = []
    for i in xrange(len(u.columns)):
        x = sm.mul(np.sign(u).iloc[:, i], axis=1)
        good = -1. * rm.rolling(lookback, min_periods=1).sum().mul(sm[x > 0].shift(lookback)).sum(axis=0).sum()
        bad = -1. * rm.rolling(lookback, min_periods=1).sum().mul(sm[x < 0].shift(lookback)).sum(axis=0).sum()
        ans.append(good - bad)
    ans = np.array(ans)
    ans /= np.sum(np.abs(ans))
    return np.sign(u).mul(ans, axis=1).sum(axis=1)


def calc_stmom_weights(rm, lookback=1):
    ans = []
    stm = get_stock_stm(rm)
    for i in xrange(52, len(rm)):
        idx = rm.index[i]
        logger.info(idx.strftime('Running %Y-%m-%d'))
        r = rm.iloc[i-52:i]
        u = get_svd_loadings(r)
        w = get_stmom_weight(u, r, stm.iloc[i-52:i], lookback)
        w.name = idx
        ans.append(w)
    ans = pd.concat(ans, axis=1).T
    x = stm.mul(ans)
    good = rm.mul(stm[x > 0].shift()).sum(axis=1)
    bad = rm.mul(stm[x < 0].shift()).sum(axis=1)
    return ans, x, good, bad


def cache_stmom_weights(universe='SMX'):
    filename = PATH + '%s_stmom.dat' % universe
    rtn, rm, vol, sx = get_dataset(universe)
    write_pickle(calc_stmom_weights(rm), filename)


def load_stmom_weights(universe='SMX'):
    filename = PATH + '%s_stmom.dat' % universe
    return load_pickle(filename)


# Lookback, timing bottom
def run_long(rtn, rm, vol, sx, x, i, start_date, end_date, style='A', bottom=True, shock=False, enhance=True, holding_period=3):
    if style == 'A':
        l = np.arange(.3, 2.01, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index]
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index]
    elif style == 'B':
        l = np.arange(.5, 3.01, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index] * np.sqrt(1. * i)
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index] * np.sqrt(52.)
    acc = rtn.cumsum()
    ax = acc.rolling(i, min_periods=1).min()
    ans = None 
    ans_high = None
    ans_low = None
    m = -1.
    mu = -1.
    df = -1.
    for high in l:
        for low in l:
            pnl = get_pnl(s1, s2, rtn, vol, high, low, acc, ax, sx, x, bottom, shock, enhance, holding_period=holding_period)
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
                    ans_high = high
                    ans_low = low
    return ans, mu, df, ans_high, ans_low

    
def estimate_reversal(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), style='A', bottom=True,
                      shock=False, enhance=True, holding_period=3):
    rtn, rm, vol, sx = get_dataset(universe)
    _, x, _, _ = load_stmom_weights(universe)
    ana = []
    params = pd.DataFrame([])
    plt.figure(figsize=(10, 7))
    for i in xrange(1, 14):
        logger.info('Lookback %d' % i)
        pnl, mu, df, ans_high, ans_low = run_long(rtn, rm, vol, sx, x, i, start_date, end_date, style, bottom, shock,
                               enhance, holding_period=holding_period)
        if pnl is not None:
            pnl.plot()
            ana.append([mu, df])
            params.loc[i, 'high'] = ans_high
            params.loc[i, 'low'] = ans_low
    plt.legend(loc='best', frameon=False)
    style_text = style + 'S' if shock else style
    style_text = style_text + 'E' if enhance else style_text
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
    write_pickle(params, PATH + '%s_%s_%s.dat' % (universe, style_text, 'Bt' if bottom else 'All'))


def run_all():
    for universe in ['SMX', 'FTSE250']:
        for style in ['A', 'B']:
            for bottom in [True, False]:
                for shock in [True, False]:
                    estimate_reversal(universe, style=style, bottom=bottom, shock=shock, enhance=True)


def load_test_results(universe, style, bottom, shock, enhance):
    style_text = style + 'S' if shock else style
    style_text = style_text + 'E' if enhance else style_text
    return load_pickle(PATH + '%s_%s_%s.dat' % (universe, style_text, 'Bt' if bottom else 'All'))


def get_long_pos(rtn, rm, vol, sx, i, start_date, high, low, style='A', bottom=True, shock=False):
    if style == 'A':
        l = np.arange(.3, 2.01, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index]
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index]
    elif style == 'B':
        l = np.arange(.5, 3.01, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index] * np.sqrt(1. * i)
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index] * np.sqrt(52.)    
    acc = rtn.cumsum()
    ax = acc.rolling(i, min_periods=1).min()
    return get_pos(s1, s2, vol, high, low, acc, ax, sx, bottom, shock)[start_date:]


def test_reversal(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), style='A', bottom=True, shock=True):
    params = load_test_results(universe, style, bottom, shock)
    style_text = style + 'S' if shock else style
    if params is not None:
        rtn, rm, vol, sx = get_dataset(universe)
        fr = rtn.rolling(4, min_periods=1).sum().shift(-4)
        plt.figure(figsize=(14, 7))
        for i in xrange(1, 13):
            plt.subplot(3, 4, i)
            logger.info('Lookback %d' % i)
            high = params.loc[i, 'high']
            low = params.loc[i, 'low']
            pos = get_long_pos(rtn, rm, vol, sx, i, start_date, high, low, style, bottom, shock)
            pr = pos.mul(tu.fit_data(fr, pos)).fillna(0.)
            tmp = rm.rolling(i + 13, min_periods=1).mean() * np.sqrt(i + 13.)
            #x = tu.get_observations(rm.rolling(i, min_periods=1).mean().shift(i).fillna(0.), pos)
            #y = tu.get_observations(pr, pos)
            #vu.binned_statistic_plot(x, y, 'mean', 7, (-3., 3.))
            #p = rm.rolling(13, min_periods=1).max().shift(i)
            tmp2 = []
            xs = np.arange(-2., 2., .5)
            for j in xs:
                a = np.mean(tu.get_observations(pr, pos[tmp <= j]))
                b = np.mean(tu.get_observations(pr, pos[tmp > j]))
                tmp2.append(np.mean(b) - np.mean(a))
            tmp2 = pd.Series(tmp2, index = xs)
            vu.bar_plot(tmp2)
            plt.title(i)
        plt.tight_layout()
        plt.savefig(PATH + '%s_%s_%s_%d_test2.png' % (universe, style_text, 'Bt' if bottom else 'All', start_date.year))
        plt.close()


def test_all():
    for universe in ['SMX', 'FTSE250']:
        for style in ['A', 'B']:
            for bottom in [True, False]:
                for shock in [True, False]:
                    test_reversal(universe, style=style, bottom=bottom, shock=shock)


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