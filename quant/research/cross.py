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


def get_volume(vol):
    v = vol.resample('W').sum()
    v = v.rolling(52, min_periods=13).median()
    v[v <= 0.] = .5
    ans = np.log(v)
    ans = ans.subtract(ans.mean(axis=1), axis=0).divide(ans.std(axis=1), axis=0)
    return ans


def get_universe_returns(universe, data_name='Returns'):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS, data_name=data_name)
    if universe == 'SMX':
        u = stocks.get_ftse_smx_universe()
    elif universe == 'FTSE250':
        u = stocks.get_ftse250_universe()
    elif universe == 'AIM':
        u = stocks.get_ftse_aim_universe()
    r = r.loc[:, r.columns.isin(u.index)]
    return r


def get_dataset(universe, max_rtn=None):
    r = get_universe_returns(universe)
    volume = get_universe_returns(universe, data_name='Volume')
    if max_rtn is None:
        if universe in ['SMX', 'AIM']:
            max_rtn = .25
        else:
            max_rtn = .2
    r, rm, v = get_returns(r, max_rtn=max_rtn)
    vol = get_volume(volume)
    return r, rm, v, vol


def get_stock_mom(rm, lookback=8):
    return rm.rolling(lookback, min_periods=1).mean() * np.sqrt(1. * lookback)


def get_momentum_weights(rtn, rm, vol, volume, stm=3):
    ans = []
    s1 = get_stock_mom(rm, stm)
    s2 = get_stock_mom(rm, 52).shift(stm)
    acc2 = rtn.cumsum()
    dd = acc2.rolling(13, min_periods=1).max() - acc2
    ltm = get_stock_mom(rm, 52).shift(3)
    wl = np.sign(ltm.subtract(ltm.mean(axis=1), axis=0)).divide(vol)
    rl = rtn.mul(wl.shift())
    z = rl.rolling(3, min_periods=1).mean()
    z = z.subtract(z.mean(axis=1), axis=0).divide(z.std(axis=1), axis=0)
    ans = 1. * (s1 <= -.5) * (s2 >= .5).divide(vol)
    ans = ans[ans > 0].ffill(limit=3)[(volume >= -.6) & (dd >= .08) & (z <= .1)]
    ans2 = 1. * ~s1.isnull() * ~s2.isnull() * (ans.fillna(0.) == 0.).divide(vol)
    good = rtn.mul(ans.shift()).sum(axis=1) / ans.sum(axis=1).shift()
    bad = rtn.mul(ans2.shift()).sum(axis=1) / ans2.sum(axis=1).shift()
    return ans, good, bad


def plot_momentum(universe='SMX'):
    rtn, rm, vol, volume = get_dataset(universe)
    ans, good, bad = get_momentum_weights(rtn, rm, vol, volume)
    plt.figure()
    good.cumsum().ffill().plot(label='Good')
    bad.cumsum().ffill().plot(label='Bad')
    (good - bad).cumsum().ffill().plot(label='Diff')
    plt.legend(loc='best', frameon=False)
    plt.title(universe, weight='bold')
    plt.tight_layout()
    plt.savefig(PATH + '%s momentum.png' % universe)
    plt.close()


# blind factor identification
def get_svd_loadings(rm):
    r = rm.subtract(rm.mean(axis=1), axis=0).fillna(0.).T
    u, s, v = np.linalg.svd(r)
    return pd.DataFrame(u, index=rm.columns).iloc[:, :len(s)]


def get_factor_returns(u, rm):
    ans = pd.DataFrame(np.dot(rm.fillna(0.), u), index=rm.index, columns=u.columns)
    return ans.divide(np.diag(np.dot(u.T, u)), axis=1) 


def get_emom_weight(u, rm):
    r = get_factor_returns(u, rm)
    f = r.mean(axis=0)
    z = r.mean(axis=0) / r.std(axis=0) * np.sqrt(52.)
    f = f * (z.abs() >= 2)
    return u.mul(f, axis=1).sum(axis=1)


def get_emom(rtn, rm, vol, volume):
    ans = []
    #w, _, _ = get_momentum_weights(rtn, rm, vol, volume)
    for i in xrange(52, len(rm)):
        idx = rm.index[i]
        logger.info(idx.strftime('Running %Y-%m-%d'))
        r = rm.iloc[i-52:i]
        u = get_svd_loadings(r)
        tmp = get_emom_weight(u, r)
        tmp.name = idx
        ans.append(tmp)
    ans = pd.concat(ans, axis=1).T
    p = w[ans > 0]
    p2 = w[ans < 0] 
    good = rtn.mul(p.shift()).sum(axis=1) / p.abs().sum(axis=1).shift()
    bad = rtn.mul(p2.shift()).sum(axis=1) / p2.abs().sum(axis=1).shift()
    tot= rtn.mul(w.shift()).sum(axis=1) / w.abs().sum(axis=1).shift()
    return ans, good, bad, tot


def show_momentum(universe='SMX'):
    rtn, rm, vol, volume = get_dataset(universe)
    stm = 3
    s1 = get_stock_mom(rm, stm)
    s2 = get_stock_mom(rm, 52).shift(stm)
    acc2 = rtn.cumsum()
    dd = acc2.rolling(13, min_periods=1).max() - acc2
    ltm = get_stock_mom(rm, 52).shift(3)
    wl = np.sign(ltm.subtract(ltm.mean(axis=1), axis=0)).divide(vol)
    rl = rtn.mul(wl.shift())
    z = rl.rolling(3, min_periods=1).mean()
    z = z.subtract(z.mean(axis=1), axis=0).divide(z.std(axis=1), axis=0)
    ans = 1. * (s1 <= -.5) * (s2 >= .5).divide(vol)
    ans = ans[ans > 0].ffill(limit=3)[(volume >= -.6) & (z <= .1) & (dd >= .08)]
    #ans, _, _ = get_momentum_weights(rtn, rm, vol, volume, 3)

    y = pd.Series([])
    for t in np.arange(.01, .15, .01):
        p = ans[dd >= t]
        p2 = ans[dd < t]
        ra = rtn.mul(p.shift()).sum(axis=1) / p.abs().sum(axis=1).shift()
        rb = rtn.mul(p2.shift()).sum(axis=1) / p2.abs().sum(axis=1).shift()
        y.loc[t] = ra.sum() - rb.sum()
    return y



def cache_momentum():
    for u in ['SMX', 'FTSE250']:
        cache_blind_momentum(u)


def main():
    cache_momentum()


if __name__ == '__main__':
    main()
