from quant.lib.main_utils import *
from quant.data import stocks
from quant.lib import portfolio_utils as pu, visualization_utils as vu, timeseries_utils as tu
from matplotlib import pyplot as plt
from scipy import stats as ss
from statsmodels import api as sm

PATH = '/home/wayne/TempWork/cross/'
make_dir(PATH)


def get_top(x, top=5):
    def _get_top(v):
        y = v.dropna()
        if y.empty:
            return v
        else:
            return y.sort_values(ascending=False).iloc[:top].loc[v.index]
    
    return x.apply(_get_top, axis=1)


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


def get_universe_returns(universe):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    if universe == 'SMX':
        u = stocks.get_ftse_smx_universe()
    elif universe == 'FTSE250':
        u = stocks.get_ftse250_universe()
    r = r.loc[:, r.columns.isin(u.index)]
    return r


def get_dataset(universe, max_rtn=None):
    r = get_universe_returns(universe)
    if max_rtn is None:
        if universe == 'SMX':
            max_rtn = .25
        else:
            max_rtn = .2
    return get_returns(r, max_rtn=max_rtn)


def get_stock_mom(rm, lookback=8):
    return rm.rolling(lookback, min_periods=1).mean() * np.sqrt(1. * lookback)


def get_momentum_weights(rtn, rm, vol, stm=3):
    ans = []
    s1 = get_stock_mom(rm, stm)
    s2 = get_stock_mom(rm, 52).shift(stm)
    acc = rm.cumsum()
    ax = acc.rolling(stm, min_periods=1).min()
    acc2 = rtn.cumsum()
    dd = acc2.rolling(13, min_periods=1).max() - acc2
    #ans = 1. * (s1 < -.5) * (s2 > .5) * (acc == ax).divide(vol)
    ans = 1. * (s1 < -.5) * (s2 > .5).divide(vol)
    ans = ans[ans > 0].ffill(limit=3)[dd >= .09]
    ans2 = 1. * ~s1.isnull() * ~s2.isnull() * (ans.fillna(0.) == 0.).divide(vol)
    good = rtn.mul(ans.shift()).sum(axis=1) / ans.sum(axis=1).shift()
    bad = rtn.mul(ans2.shift()).sum(axis=1) / ans2.sum(axis=1).shift()
    return ans, good, bad


def plot_momentum(universe='SMX'):
    rtn, rm, vol = get_dataset(universe)
    ans, good, bad = get_momentum_weights(rtn, rm, vol)
    plt.figure()
    good.cumsum().ffill().plot(label='Good')
    bad.cumsum().ffill().plot(label='Bad')
    plt.legend(loc='best', frameon=False)
    plt.title(universe, weight='bold')
    plt.tight_layout()
    #plt.savefig(PATH + '%s momentum.png' % universe)
    #plt.close()

# blind factor identification
def get_svd_loadings(rm):
    r = rm.subtract(rm.mean(axis=1), axis=0).fillna(0.).T
    u, s, v = np.linalg.svd(r)
    return pd.DataFrame(u, index=rm.columns).iloc[:, :len(s)]


def get_emom_weight(u, rm, ltm):
    r = []
    x = sm.add_constant(u)
    for idx in rm.index:
        y = rm.loc[idx].fillna(0.)
        lm = sm.OLS(y, x)
        m = lm.fit()
        tmp = m.params.iloc[1:]
        tmp.name = idx
        r.append(tmp)
    r = pd.concat(r, axis=1).T
    s = r.sum(axis=0)
    s /= s.abs().sum()
    u2 = u.mul(s, axis=1)
    #s = rm[ltm.shift() < 0].sum(axis=0) - rm[ltm.shift() > 0].sum(axis=0)
    #u3 = (u2 > 0).mul(s, axis=0).sum(axis=0)
    #u3[u3 < 0] = 0.
    return np.sign(u2.sum(axis=1))


def get_emom(rtn, rm, vol):
    ans = []
    ltm = get_stock_mom(rm, 52)
    for i in xrange(52, len(rm)):
        idx = rm.index[i]
        logger.info(idx.strftime('Running %Y-%m-%d'))
        r = rm.iloc[i-52:i]
        lm = ltm.iloc[i-52:i]
        u = get_svd_loadings(r)
        w = get_emom_weight(u, r, lm)
        w.name = idx
        ans.append(w)
    ans = pd.concat(ans, axis=1).T.divide(vol)[~rm.isnull()]
    w, _, _ = get_momentum_weights(rtn, rm, vol)
    p = w[ans > 0]
    p2 = w[ans < 0]
    #p = ans[ans > 0].ffill(limit=3)
    #p2 = 1. * ~rm.isnull() * (p.fillna(0.) == 0.).divide(vol)
    good = rtn.mul(p.shift()).sum(axis=1) / p.sum(axis=1).shift()
    bad = rtn.mul(p2.shift()).sum(axis=1) / p2.sum(axis=1).shift()
    return ans, good, bad


def cache_blind_momentum(universe='SMX'):
    filename = PATH + '%s_bmom.dat' % universe
    rtn, rm, vol = get_dataset(universe)
    write_pickle(get_blind_momentum(rtn, rm, vol), filename)


def load_blind_momentum(universe='SMX'):
    filename = PATH + '%s_bmom.dat' % universe
    return load_pickle(filename)


def calc_res_momentum(rm, u):
    r = []
    x = sm.add_constant(u)
    for idx in rm.index:
        y = rm.loc[idx].fillna(0.)
        lm = sm.OLS(y, x)
        m = lm.fit()
        tmp = m.resid
        tmp.name = idx
        r.append(tmp)
    r = pd.concat(r, axis=1).T[~rm.isnull()]
    stm = r.iloc[-3:].mean(axis=0) * np.sqrt(3.)
    ltm = r.iloc[:-3].mean(axis=0) * np.sqrt(len(r) - 3.)
    stm.name = rm.index[-1]
    ltm.name = rm.index[-1]
    return stm, ltm


def get_res_momentum(rtn, rm, vol, lookback=4, factors=10):
    s1 = []
    s2 = []
    for i in xrange(52, len(rm)):
        idx = rm.index[i]
        logger.info(idx.strftime('Running %Y-%m-%d'))
        r = rm.iloc[i-52:i]
        u = get_svd_loadings(r)
        s, l = calc_res_momentum(r, u.iloc[:, factors])
        s1.append(s)
        s2.append(l)
    s1 = pd.concat(s1, axis=1).T
    s2 = pd.concat(s2, axis=1).T
    ans = 1. * (s1 < -.5) * (s2 > .5).divide(vol)
    ans = ans[ans > 0].ffill(limit=lookback-1)
    ans2 = 1. * ~s1.isnull() * ~s2.isnull() * (ans.fillna(0.) == 0.).divide(vol)
    good = rtn.mul(ans.shift()).sum(axis=1) / ans.sum(axis=1).shift()
    bad = rtn.mul(ans2.shift()).sum(axis=1) / ans2.sum(axis=1).shift()
    return ans, good, bad

        
        
def get_ltmom_weight(u, rm, lm, lookback=1):
    ans = []
    for i in xrange(len(u.columns)):
        x = lm.mul(np.sign(u).iloc[:, i], axis=1)
        g = lm[x > 0].shift(lookback)
        good = rm.rolling(lookback, min_periods=1).sum().mul(g).sum(axis=1).divide(g.abs().sum(axis=1), axis=0)
        b = lm[x < 0].shift(lookback)
        bad = rm.rolling(lookback, min_periods=1).sum().mul(b).sum(axis=1).divide(b.abs().sum(axis=1), axis=0)
        t = (good.mean() - bad.mean()) / np.sqrt(good.var() / len(good) + bad.var() / len(bad))
        if np.abs(t) >= 2:
            s = good.sum() - bad.sum()
        else:
            s = 0.
        ans.append(s)
    ans = np.array(ans)
    if np.sum(np.abs(ans)) > 0:
        ans /= np.sum(np.abs(ans))
    return np.sign(u).mul(ans, axis=1).sum(axis=1)


def get_long_weight(u, rm, sm, lookback=1):
    ans = []
    for i in xrange(len(u.columns)):
        x = sm.mul(np.sign(u).iloc[:, i], axis=1)
        g = sm[x < 0].shift(lookback)
        good = rm.rolling(lookback, min_periods=1).sum().mul(g).sum(axis=1).divide(g.abs().sum(axis=1), axis=0)
        b = sm[x > 0].shift(lookback)
        bad = rm.rolling(lookback, min_periods=1).sum().mul(b).sum(axis=1).divide(b.abs().sum(axis=1), axis=0)
        #t = (good.mean() - bad.mean()) / np.sqrt(good.var() / len(good) + bad.var() / len(bad))
        #if np.abs(t) >= 2:
        s = good.sum() - bad.sum()
        #else:
        #    s = 0.
        ans.append(s)
    ans = np.array(ans)
    if np.sum(np.abs(ans)) > 0:
        ans /= np.sum(np.abs(ans))
    return np.sign(u).mul(ans, axis=1).sum(axis=1)


def calc_momentum_weights(rtn, rm, vol, lookback=4):
    ans = []
    s1 = get_stock_mom(rm, 3)
    s2 = get_stock_mom(rm, 52).shift(3)
    ltm = 1. * (s1 < -1) * (s2 > -1).divide(vol)
    for i in xrange(52, len(rm)):
        idx = rm.index[i]
        logger.info(idx.strftime('Running %Y-%m-%d'))
        r = rm.iloc[i-52:i]
        r2 = rtn.iloc[i-52:i]
        u = get_svd_loadings(r)
        w = get_long_weight(u, r2, ltm.iloc[i-52:i], lookback)
        w.name = idx
        ans.append(w)
    ans = pd.concat(ans, axis=1).T
    good = rtn.mul(ltm[ans > 0].shift()).sum(axis=1) / ltm[ans > 0].abs().sum(axis=1).shift()
    bad = rtn.mul(ltm[ans < 0].shift()).sum(axis=1) / ltm[ans < 0].abs().sum(axis=1).shift()
    return ans, good, bad


def run_regression(y, x):
    x = sm.add_constant(x.fillna(0.))
    lm = sm.OLS(y.fillna(0.), x)
    return lm.fit()

      
def get_neutral_returns(rm, stm):
    ans = []
    for idx in rm.index:
        y = rm.loc[idx].dropna()
        if not y.empty:
            x = stm.loc[idx, y.index]
            if (~x.isnull()).any():
                m = run_regression(y, x)
                tmp = m.resid
                tmp.name = idx
                ans.append(tmp)
            else:
                ans.append(y)
    ans = pd.concat(ans, axis=1).T
    return ans[~rm.isnull()]


def show_momentum(universe='SMX'):
    rtn, rm, vol = get_dataset(universe)
    acc = rtn.cumsum()
    s = acc.rolling(13, min_periods=1).max() - acc
    ans, _, _ = get_momentum_weights(rtn, rm, vol, 9)
    y = pd.Series([])
    for x in np.arange(.05, .15, .01):
        p = ans[s >= x]
        p2 = ans[s < x]
        r = rtn.mul(p.shift()).sum(axis=1) / p.abs().sum(axis=1).shift()
        r2 = rtn.mul(p2.shift()).sum(axis=1) / p2.abs().sum(axis=1).shift()
        y.loc[x] = r.sum() - r2.sum()
    return y
    
    x = .09
    p = ans[s >= x]
    p2 = ans[s < x]
    r = rtn.mul(p.shift()).sum(axis=1) / p.abs().sum(axis=1).shift()
    r2 = rtn.mul(p2.shift()).sum(axis=1) / p2.abs().sum(axis=1).shift()
    r3 = rtn.mul(ans.shift()).sum(axis=1) / ans.abs().sum(axis=1).shift()
    plt.figure()
    r.cumsum().plot(label='A')
    r2.cumsum().plot(label='B')
    r3.cumsum().plot(label='Orig')
    plt.legend(loc='best', frameon=False)
    plt.title(universe, weight='bold')
    plt.tight_layout()
    plt.savefig(PATH + '%s momentum chart.png' % universe)


def cache_momentum():
    for u in ['SMX', 'FTSE250']:
        cache_blind_momentum(u)


def main():
    cache_momentum()


if __name__ == '__main__':
    main()
