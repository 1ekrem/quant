from quant.lib.main_utils import *
from quant.data import stocks
from quant.lib import portfolio_utils as pu, visualization_utils as vu, timeseries_utils as tu
from matplotlib import pyplot as plt
from scipy import stats as ss
from quant.lib.timeseries_utils import fit_data
from statsmodels import api as sm


PATH = '/home/wayne/TempWork/cross/'
make_dir(PATH)


def _get_first(x):
    idx = x.index[x.abs() > 0]
    if len(idx) > 0:
        ans = x.copy()
        ans[ans.index < idx[0]] = np.nan
        return ans
    else:
        return x * np.nan


def get_returns(r, max_rtn=.25):
    rc = r[r.abs() < max_rtn]
    rtn = r.resample('W').sum().apply(_get_first, axis=0)
    rtn2 = rc.resample('W').sum().apply(_get_first, axis=0)
    w = rtn.abs()
    vol = w[w > 0].rolling(52, min_periods=13).median()
    vol[vol < 5e-3] = 5e-3
    vol2 = vol.copy()
    vol2[vol2 < .02] = .02
    rv = rtn2.divide(vol)
    rm = rv.subtract(rv.mean(axis=1), axis=0)
    return rtn, rm, vol2


def get_dataset(universe, max_rtn=None):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    if universe == 'SMX':
        u = stocks.get_ftse_smx_universe()
    elif universe == 'FTSE250':
        u = stocks.get_ftse250_universe()
    r = r.loc[:, r.columns.isin(u.index)]
    if max_rtn is None:
        if universe == 'SMX':
            max_rtn = .25
        else:
            max_rtn = .2
    return get_returns(r, max_rtn=max_rtn)


def get_pos(s1, s2, vol, high, low, x, enhance=True):
    pos = 1. * ((s1 <= -low) & (s2 >= high))
    if enhance:
        pos = pos[x > 0]
    pos = pos.divide(vol)
    return pos


def get_pnl(s1, s2, r, vol, high, low, x, enhance=True, holding_period=3):
    p = get_pos(s1, s2, vol, high, low, x, enhance)
    if p.abs().sum(axis=1).sum() > 0:
        p = p[p.abs() > 0].ffill(limit=holding_period)
        g = p.abs().sum(axis=1).fillna(0.)
        pnl = r.mul(p.shift()).sum(axis=1)
        pnl = pnl / g.shift()
        pnl = _get_first(pnl).fillna(0.)
        return pnl
    else:
        return None


# blind factor identification
def get_svd_loadings(rm):
    r = rm.subtract(rm.mean(axis=1), axis=0).fillna(0.).T
    u, s, v = np.linalg.svd(r)
    return pd.DataFrame(u, index=rm.columns).iloc[:, :len(s)]


def get_stock_mom(rm, lookback=8):
    m = rm.rolling(lookback, min_periods=1).sum()
    return np.sign(m.subtract(m.mean(axis=1), axis=0))


def get_stmom_weight(u, rm, sm, lookback=1):
    ans = []
    for i in xrange(len(u.columns)):
        x = sm.mul(np.sign(u).iloc[:, i], axis=1)
        g = sm[x > 0].shift(lookback)
        good = -1. * rm.rolling(lookback, min_periods=1).sum().mul(g).sum(axis=1).divide(g.abs().sum(axis=1), axis=0).sum()
        b = sm[x < 0].shift(lookback)
        bad = -1. * rm.rolling(lookback, min_periods=1).sum().mul(b).sum(axis=1).divide(b.abs().sum(axis=1), axis=0).sum()
        ans.append(good - bad)
    ans = np.array(ans)
    ans /= np.sum(np.abs(ans))
    return np.sign(u).mul(ans, axis=1).sum(axis=1)


def get_ltmom_weight(u, rm, lm, lookback=1):
    ans = []
    for i in xrange(len(u.columns)):
        x = lm.mul(np.sign(u).iloc[:, i], axis=1)
        g = lm[x > 0].shift(lookback)
        good = rm.rolling(lookback, min_periods=1).sum().mul(g).sum(axis=1).divide(g.abs().sum(axis=1), axis=0).sum()
        b = lm[x < 0].shift(lookback)
        bad = rm.rolling(lookback, min_periods=1).sum().mul(b).sum(axis=1).divide(b.abs().sum(axis=1), axis=0).sum()
        ans.append(good - bad)
    ans = np.array(ans)
    ans /= np.sum(np.abs(ans))
    return np.sign(u).mul(ans, axis=1).sum(axis=1)


def calc_momentum_weights(rtn, rm, vol, lookback=4):
    ans = []
    ans2 = []
    stm = get_stock_mom(rm, 3).divide(vol)
    ltm = get_stock_mom(rm, 52).shift(3).divide(vol)
    for i in xrange(52, len(rm)):
        idx = rm.index[i]
        logger.info(idx.strftime('Running %Y-%m-%d'))
        r = rm.iloc[i-52:i]
        r2 = rtn.iloc[i-52:i]
        u = get_svd_loadings(r)
        w = get_stmom_weight(u, r2, stm.iloc[i-52:i], lookback)
        w.name = idx
        ans.append(w)
        w2 = get_ltmom_weight(u, r2, ltm.iloc[i-52:i], lookback)
        w2.name = idx
        ans2.append(w2)
    ans = pd.concat(ans, axis=1).T
    ans2 = pd.concat(ans2, axis=1).T
    ans = stm.mul(ans)
    ans2 = ltm.mul(ans2)
    good = rtn.mul(stm[ans > 0].shift()).sum(axis=1) / stm[ans > 0].abs().sum(axis=1).shift()
    bad = rtn.mul(stm[ans < 0].shift()).sum(axis=1) / stm[ans < 0].abs().sum(axis=1).shift()
    good2 = rtn.mul(ltm[ans2 > 0].shift()).sum(axis=1) / ltm[ans > 0].abs().sum(axis=1).shift()
    bad2 = rtn.mul(ltm[ans2 < 0].shift()).sum(axis=1) / ltm[ans < 0].abs().sum(axis=1).shift()
    return ans, ans2, good, bad, good2, bad2


def get_neutral_returns(rm, stm):
    ans = []
    for idx in rm.index:
        y = rm.loc[idx].dropna()
        if not y.empty:
            x = stm.loc[idx, y.index]
            if (~x.isnull()).any():
                x = sm.add_constant(x.fillna(0.))
                lm = sm.OLS(y, x)
                m = lm.fit()
                tmp = m.resid
                tmp.name = idx
                ans.append(tmp)
            else:
                ans.append(y)
    ans = pd.concat(ans, axis=1).T
    return ans[~rm.isnull()]


def get_blind_momentum(rm, u, lookback=4):
    r = []
    r2 = []
    x = sm.add_constant(u)
    for idx in rm.index:
        y = rm.loc[idx].fillna(0.)
        lm = sm.OLS(y, x)
        m = lm.fit()
        tmp = m.params
        tmp.name = idx
        r.append(tmp)
        tmp2 = m.resid
        tmp2.name = idx
        r2.append(tmp2)
    r = pd.concat(r, axis=1).T.loc[:, u.columns]
    r2 = pd.concat(r2, axis=1).T
    stm = u.mul(r.iloc[-lookback:].mean(axis=0), axis=1).sum(axis=1)
    ltm = u.mul(r.iloc[:-lookback].mean(axis=0), axis=1).sum(axis=1)
    stm.name = rm.index[-1]
    ltm.name = rm.index[-1]
    stm2 = r2.iloc[-lookback:].mean(axis=0)
    ltm2 = r2.iloc[:-lookback].mean(axis=0)
    stm2.name = rm.index[-1]
    ltm2.name = rm.index[-1]
    return stm, ltm, stm2, ltm2


def calc_blind_momentum(rtn, rm, vol):
    m_s = rm.rolling(52, min_periods=1).sum().shift(4)
    rs = get_neutral_returns(rm, m_s)
    stm = get_stock_mom(rs, 3).divide(vol)
    ltm = get_stock_mom(rs, 52).shift(3).divide(vol)
    s = rtn.mul(stm.shift()).sum(axis=1) / stm.abs().sum(axis=1).shift()
    l = rtn.mul(ltm.shift()).sum(axis=1) / ltm.abs().sum(axis=1).shift()
    return m_s, stm, ltm, s, l


def cache_momentum_weights(universe='SMX'):
    filename = PATH + '%s_mom.dat' % universe
    rtn, rm, vol = get_dataset(universe)
    write_pickle(calc_momentum_weights(rtn, rm, vol), filename)


def load_momentum_weights(universe='SMX'):
    filename = PATH + '%s_mom.dat' % universe
    return load_pickle(filename)


def plot_momentum_enhancement(universe='SMX'):
    a, a2, g, b, g2, b2 = load_momentum_weights(universe)
    plt.figure()
    g.cumsum().plot(label='Good Rev')
    b.cumsum().plot(label='Bad Rev')
    g2.cumsum().plot(label='Good Mom')
    b2.cumsum().plot(label='Bad Mom')
    plt.legend(loc='best', frameon=False)
    plt.title(universe, weight='bold')
    plt.tight_layout()
    plt.savefig(PATH + '%s momentum.png' % universe)
    plt.close()


def cache_blind_momentum(universe='SMX'):
    filename = PATH + '%s_bmom.dat' % universe
    rtn, rm, vol = get_dataset(universe)
    write_pickle(calc_blind_momentum(rtn, rm, vol), filename)


def load_blind_momentum(universe='SMX'):
    filename = PATH + '%s_bmom.dat' % universe
    return load_pickle(filename)


def plot_blind_momentum(universe='SMX'):
    a, a2, g, b, g2, b2 = load_blind_momentum(universe)
    plt.figure()
    g.cumsum().plot(label='F Rev')
    b.cumsum().plot(label='F Mom')
    g2.cumsum().plot(label='S Rev')
    b2.cumsum().plot(label='S Mom')
    plt.legend(loc='best', frameon=False)
    plt.title(universe, weight='bold')
    plt.tight_layout()
    plt.savefig(PATH + '%s F momentum.png' % universe)
    plt.close()


# Lookback, timing bottom
def run_long(rtn, rm, vol, x, i, start_date, end_date, style='A', enhance=True, holding_period=3):
    run_set = []
    if style in ['A', 'C']:
        l = np.arange(0.2, 1.21, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index]
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index]
        for high in l:
            for low in l:
                if high <= low:
                    run_set.append((high, low))
    elif style in ['B', 'D']:
        l = np.arange(1., 2.01, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index] * np.sqrt(1. * i)
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index] * np.sqrt(52.)
        for high in l:
            for low in l:
                if high <= low:
                    run_set.append((high, low))
    acc = rtn.cumsum()
    ax = acc.rolling(4, min_periods=1).min()
    s1 = s1[acc == ax]
    s2 = s2[acc == ax]
    ans = None 
    ans_high = None
    ans_low = None
    m = -1.
    mu = -1.
    df = -1.
    for high, low in run_set:
        pnl = get_pnl(s1, s2, rtn, vol, high, low, x, enhance, holding_period=holding_period)
        if pnl is not None:
            pnl = pnl[start_date:]
            tot = pnl.mean() - .5 * pnl.std() ** 2
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

    
def estimate_reversal(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), style='A',
                      enhance=True, holding_period=3):
    rtn, rm, vol = get_dataset(universe)
    x, x2, _, _, _, _ = load_momentum_weights(universe)
    x = x[x2 > 0]
    ana = []
    params = pd.DataFrame([])
    plt.figure(figsize=(10, 7))
    for i in xrange(1, 14):
        logger.info('Lookback %d' % i)
        pnl, mu, df, ans_high, ans_low = run_long(rtn, rm, vol, x, i, start_date, end_date, style,
                               enhance, holding_period=holding_period)
        if pnl is not None:
            pnl.plot()
            ana.append([mu, df])
            params.loc[i, 'high'] = ans_high
            params.loc[i, 'low'] = ans_low
    plt.legend(loc='best', frameon=False)
    style_text = style + 'E' if enhance else style
    plt.title('%s %s' % (universe, style_text), weight='bold')
    plt.tight_layout()
    plt.savefig(PATH + '%s_%s_%d.png' % (universe, style_text, start_date.year))
    plt.close()
    plt.figure()
    vu.bar_plot(pd.DataFrame(ana, index=np.arange(1, 14), columns=['Mean', 'Decay']).T)
    plt.plot(np.arange(len(ana)) * 3 + 1, np.sum(ana, axis=1), marker='s', color='black')
    plt.axhline(.35, color='grey', ls='--')
    plt.title('%s %s' % (universe, style_text), weight='bold')
    plt.tight_layout()
    plt.savefig(PATH + '%s_%s_%d_decay.png' % (universe, style_text, start_date.year))
    plt.close()
    write_pickle(params, PATH + '%s_%s.dat' % (universe, style_text))


def run_all():
    for universe in ['SMX', 'FTSE250']:
        for style in ['C', 'D']:
            for enhance in [True, False]:
                estimate_reversal(universe, style=style, enhance=enhance)


def load_params(universe, style, enhance):
    style_text = style + 'E' if enhance else style
    return load_pickle(PATH + '%s_%s.dat' % (universe, style_text))


def get_long_pos(rtn, rm, vol, x, i, start_date, high, low, style='A', enhance=True):
    if style == 'A':
        l = np.arange(0.2, 1.21, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index]
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index]
    elif style == 'B':
        l = np.arange(1., 2.01, .1)
        s1 = rm.rolling(i, min_periods=1).mean().loc[rtn.index] * np.sqrt(1. * i)
        s2 = rm.rolling(52, min_periods=13).mean().shift(i).loc[rtn.index] * np.sqrt(52.)
    acc = rtn.cumsum()
    ax = acc.rolling(i, min_periods=1).min()
    s1 = s1[acc == ax]
    s2 = s2[acc == ax]
    return get_pos(s1, s2, vol, high, low, x, enhance)[start_date:]


def test_reversal(universe='SMX', start_date=dt(2009, 1, 1), end_date=dt(2015, 12, 31), style='A', enhance=True):
    params = load_params(universe, style, enhance)
    style_text = style + 'E' if enhance else style
    if params is not None:
        rtn, rm, vol, s = get_dataset(universe)
        x, x2, _, _, _, _ = load_momentum_weights(universe)
        x = x[x2 > 0]
        fr = rtn.rolling(4, min_periods=1).sum().shift(-4)
        plt.figure(figsize=(14, 7))
        for i in xrange(1, 13):
            plt.subplot(3, 4, i)
            logger.info('Lookback %d' % i)
            high = params.loc[i, 'high']
            low = params.loc[i, 'low']
            pos = get_long_pos(rtn, rm, vol, x, i, start_date, high, low, style, enhance)
            pr = pos.mul(tu.fit_data(fr, pos)).fillna(0.)
            xx = tu.get_observations(s.rolling(i, min_periods=1).min().fillna(0.), pos)
            y = tu.get_observations(pr, pos)
            vu.binned_statistic_plot(xx, y, 'mean', 7, (-.2, 0.))
            #p = rm.rolling(13, min_periods=1).max().shift(i)
            #tmp2 = []
            #xs = np.arange(-.2, .2, .5)
            #for j in xs:
            #    a = np.mean(tu.get_observations(pr, pos[tmp <= j]))
            #    b = np.mean(tu.get_observations(pr, pos[tmp > j]))
            #    tmp2.append(np.mean(b) - np.mean(a))
            #tmp2 = pd.Series(tmp2, index = xs)
            #vu.bar_plot(tmp2)
            plt.title(i)
        plt.tight_layout()
        plt.savefig(PATH + '%s_%s_%d_test.png' % (universe, style_text, start_date.year))
        plt.close()


def test_all():
    for universe in ['SMX', 'FTSE250']:
        for style in ['A', 'B']:
            for enhance in [True, False]:
                test_reversal(universe, style=style, enhance=enhance)


def cache_momentum():
    for u in ['SMX', 'FTSE250']:
        cache_momentum_weights(u)


def main():
    cache_momentum()


if __name__ == '__main__':
    main()
