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
            return y.sort_values(ascending=False).iloc[:top].reindex(v.index)
    
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
    u = stocks.load_universe(universe)
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


def load_fundamental_changes(data_type, u):
    data = stocks.load_financial_data(data_type)
    if data is None:
        return None
    else:
        data = data.reindex(u.index, axis=1)
        today = dt.today()
        if today not in data.index:
            data.loc[today] = np.nan
        data = data.resample('M').last()
        return tu.get_calendar_df(data)
    

def load_financials(universe='SMX'):
    financials = {}
    ids = ['Turnover', 'Pretax Profit', 'Operating Profit', 'EPS Diluted',
           'revenue', 'profit', 'ebit', 'eps', 'ebitda', 'fcf']
    u = stocks.load_universe(universe)
    for x in ids:
        data = load_fundamental_changes(x, u)
        data2 = load_fundamental_changes('Interim ' + x, u)
        if data2 is not None:
            data = data.add(2. * data2, fill_value=0.)
        financials[x] = data.shift()
    return financials


def get_financials_overall_score(financials):
    ans = None
    for v in financials.values():
        tmp = v.ffill()
        tmp2 = 1. * (tmp > 0) - 1. * (tmp < 0)
        if ans is None:
            ans = tmp2
        else:
            ans = ans.add(tmp2, fill_value=0.)
    return ans


def get_stock_mom(rm, lookback=8):
    return rm.rolling(lookback, min_periods=1).mean() * np.sqrt(1. * lookback)


def get_step_positions(input1, input2, vol, ns, f=None, f2=None, holding=3):
    s1 = input1 if f is None else input1[f > 0]
    s2 = input2 if f is None else input2[f > 0]
    fast = get_top(s1, ns * 2)
    p = get_top(s2[~fast.isnull()], ns)
    ans = (1. / vol)[~p.isnull()]
    if holding > 0:
        ans = ans.ffill(limit=holding)
    if f2 is not None:
        ans = ans[f2 > 0]
    return ans


def get_fast_weights(rtn, rm, vol, volume, stm=3, ns=3, holding=0):
    s1 = -1. * get_stock_mom(rm, stm)
    s2 = get_stock_mom(rm, 52).shift(stm)
    s3 = get_stock_mom(rm, 52)
    acc2 = rtn.cumsum()
    dd = acc2.rolling(13, min_periods=1).max() - acc2
    base = 1. * (volume >= -.6) * (dd >= .05) * (s1 >= 1.6) * (s3 >= -.5)
    ans = get_step_positions(s1, s2, vol, ns, base, holding=holding)
    ans2 = 1. * ~s1.isnull() * ~s2.isnull() * (ans.fillna(0.) == 0.).divide(vol)
    good = rtn.mul(ans.shift()).sum(axis=1) / ans.sum(axis=1).shift()
    bad = rtn.mul(ans2.shift()).sum(axis=1) / ans2.sum(axis=1).shift()
    return ans, good, bad


def get_fast_fundamental_weights(rtn, rm, vol, volume, score, stm=3, ns=3, holding=0):
    s1 = -1. * get_stock_mom(rm, stm)
    s2 = get_stock_mom(rm, 52).shift(stm)
    s3 = get_stock_mom(rm, 52)
    acc2 = rtn.cumsum()
    dd = acc2.rolling(13, min_periods=1).max() - acc2
    base = 1. * (volume >= -.6) * (dd >= .05) * (s1 >= 1.6) * (s3 >= -.5)
    ans = get_step_positions(s1, score, vol, ns, base, None, holding=holding)
    ans2 = 1. * ~s1.isnull() * ~s2.isnull() * (ans.fillna(0.) == 0.).divide(vol)
    good = rtn.mul(ans.shift()).sum(axis=1) / ans.sum(axis=1).shift()
    bad = rtn.mul(ans2.shift()).sum(axis=1) / ans2.sum(axis=1).shift()
    return ans, good, bad


def get_slow_weights(rtn, rm, vol, volume, stm=4, ns=12, holding=3):
    s1 = -1. * get_stock_mom(rm, stm)
    s2 = get_stock_mom(rm, 52).shift(stm)
    s3 = get_stock_mom(rm, 52)
    acc2 = rtn.cumsum()
    dd = acc2.rolling(13, min_periods=1).max() - acc2
    base = 1. * (volume >= -.6)
    base2 = 1. * (s1 >= 1.6) * (s3 >= .1)
    ans = get_step_positions(s2, s1, vol, ns, base, base2, holding=holding)
    ans2 = 1. * ~s1.isnull() * ~s2.isnull() * (ans.fillna(0.) == 0.).divide(vol)
    good = rtn.mul(ans.shift()).sum(axis=1) / ans.sum(axis=1).shift()
    bad = rtn.mul(ans2.shift()).sum(axis=1) / ans2.sum(axis=1).shift()
    return ans, good, bad


def get_slow_fundamental_weights(rtn, rm, vol, volume, score, stm=4, ns=12, holding=3):
    s1 = -1. * get_stock_mom(rm, stm)
    s2 = get_stock_mom(rm, 52).shift(stm)
    s3 = get_stock_mom(rm, 52)
    base = 1. * (volume >= -.6)
    base2 = 1. * (s1 >= 1.3) * (s3 >= -.5)
    ans = get_step_positions(s2, score, vol, ns, base, base2, holding=holding)
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
    fr = get_factor_returns(u, rm)
    return u.mul(fr.iloc[-13:].mean(axis=0) - fr.mean(axis=0) * np.sqrt(52. / 13.), axis=1).sum(axis=1)


def get_emom(rtn, rm, vol, volume):
    ans = []
    w, _, _ = get_momentum_weights(rtn, rm, vol, volume)
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
    return ans, good.fillna(0.), bad.fillna(0.), tot


def get_portfolio_returns(ans, rtn):
    ra = rtn.mul(ans.shift()).sum(axis=1) / ans.abs().sum(axis=1).shift()
    return ra.fillna(0.)


def show_portfolio_size(universe='SMX'):
    rtn, rm, vol, volume = get_dataset(universe)
    data = load_financials(universe)
    data = tu.resample(get_financials_overall_score(data), rtn)
    ans = pd.DataFrame([])
    ans2 = pd.DataFrame([])
    for s in np.arange(1, 21):
        _, g, _ = get_fast_weights(rtn, rm, vol, volume, ns=s)
        _, g2, _ = get_fast_fundamental_weights(rtn, rm, vol, volume, data, ns=s)
        _, g3, _ = get_slow_weights(rtn, rm, vol, volume, ns=s)
        _, g4, _ = get_slow_fundamental_weights(rtn, rm, vol, volume, data, ns=s)
        ans.loc[s, 'Fast'] = g.sum()
        ans.loc[s, 'Fast F'] = g2.sum()
        ans.loc[s, 'Slow'] = g3.sum()
        ans.loc[s, 'Slow F'] = g4.sum()
        ans2.loc[s, 'Fast'] = g[dt(2014,1,1):].sum()
        ans2.loc[s, 'Fast F'] = g2[dt(2014,1,1):].sum()
        ans2.loc[s, 'Slow'] = g3[dt(2014,1,1):].sum()
        ans2.loc[s, 'Slow F'] = g4[dt(2014,1,1):].sum()
    return ans, ans2
        


def test_momentum(universe='SMX'):
    rtn, rm, vol, volume = get_dataset(universe)
    stm = 4
    ltm = 52
    ns = 12
    s1 = -1. * get_stock_mom(rm, stm)
    s2 = get_stock_mom(rm, 52).shift(stm)
    s3 = get_stock_mom(rm, 52)
    acc2 = rtn.cumsum()
    dd = acc2.rolling(13, min_periods=1).max() - acc2
    base = 1. * (volume >= -.6)# * (dd >= .08) * (s1 >= 1.6) * (s3 >= -.4) 
    base2 = 1. * (s1 >= .9) * (s3 >= .4)

    financials = load_financials(universe)
#     plt.figure()
#     y = pd.DataFrame([])
#     z = pd.DataFrame([])
#     c = pd.DataFrame([])
#     #x = np.arange(0, .15, .01)
#     ans = get_step_positions(s1, s2, vol, ns, base, None, holding=0)
#     ans2 = get_step_positions(s2, s1, vol, ns, base, None, holding=0)
#     ra = get_portfolio_returns(ans, rtn)
#     rb = get_portfolio_returns(ans2, rtn)
#     y.loc['orig', 'Fast'] = ra.sum()
#     y.loc['orig', 'Slow'] = rb.sum()
#     z.loc['orig', 'Fast'] = ra[dt(2012, 1, 1):].sum()
#     z.loc['orig', 'Slow'] = rb[dt(2012, 1, 1):].sum()
#     c.loc['orig', 'Fast'] = (ans > 0)[dt(2012, 1, 1):].sum(axis=1).mean()
#     c.loc['orig', 'Slow'] = (ans2 > 0)[dt(2012, 1, 1):].sum(axis=1).mean()
#     rb[dt(2012, 1, 1):].cumsum().plot(label='orig')
#     for k, v in financials.iteritems():
#         x = tu.resample(v, rtn)
#         f = 1. * (x > 0) * (base > 0)
#         ans = get_step_positions(s1, s2, vol, ns, f, None, holding=0)
#         ans2 = get_step_positions(s2, s1, vol, ns, f, None, holding=0)
#         ra = get_portfolio_returns(ans, rtn)
#         rb = get_portfolio_returns(ans2, rtn)
#         y.loc[k, 'Fast'] = ra.sum()
#         y.loc[k, 'Slow'] = rb.sum()
#         z.loc[k, 'Fast'] = ra[dt(2012, 1, 1):].sum()
#         z.loc[k, 'Slow'] = rb[dt(2012, 1, 1):].sum()
#         c.loc[k, 'Fast'] = (ans > 0)[dt(2012, 1, 1):].sum(axis=1).mean()
#         c.loc[k, 'Slow'] = (ans2 > 0)[dt(2012, 1, 1):].sum(axis=1).mean()
#         ra[dt(2012, 1, 1):].cumsum().plot(label=k)
#     return y, z, c

#     u = stocks.load_universe(universe)
#     data = load_fundamental_changes('EPS Diluted', u)
#     data2 = load_fundamental_changes('Interim ' + 'EPS Diluted', u)
#     if data2 is not None:
#         data = data.add(2. * data2, fill_value=0.)
#     data = tu.resample(cs(data, 2).ffill().shift(), rtn)
    data = get_financials_overall_score(financials)
    data = tu.resample(data, rtn)

    y = pd.DataFrame([])
    z = pd.DataFrame([])
    c = pd.DataFrame([])
    x = np.arange(1, 15)
    for k in x:
        s1 = -1. * get_stock_mom(rm, k)
        s2 = get_stock_mom(rm, 52).shift(k)
        s3 = get_stock_mom(rm, 52)
        f = 1. * (s1 >= 1.3) * (s3 >= -.5)
        #f2 = 1. * (s3 >= k) * (s1 >= .9)
        ans = get_step_positions(s1, data, vol, ns, base, f, holding=3)
        ans2 = get_step_positions(s2, data, vol, ns, base, f, holding=3)
        ra = get_portfolio_returns(ans, rtn)
        rb = get_portfolio_returns(ans2, rtn)
        y.loc[k, 'Fast'] = ra.sum()
        y.loc[k, 'Slow'] = rb.sum()
        z.loc[k, 'Fast'] = ra[dt(2012, 1, 1):].sum()
        z.loc[k, 'Slow'] = rb[dt(2012, 1, 1):].sum()
        c.loc[k, 'Fast'] = (ans > 0)[dt(2012, 1, 1):].sum(axis=1).mean()
        c.loc[k, 'Slow'] = (ans2 > 0)[dt(2012, 1, 1):].sum(axis=1).mean()
    return y, z, c
    
    ans = get_step_positions(s1, s2, vol, ns, base, base2)
    ans2 = get_step_positions(s2, s1, vol, ns, base, base2)
    #ans3 = get_step_positions(s1, s2, vol, ns, base, f)
    #ans4 = get_step_positions(s2, s1, vol, ns, base, f)

    ra = get_portfolio_returns(ans, rtn)
    rb = get_portfolio_returns(ans2, rtn)
    #rc = get_portfolio_returns(ans3, rtn)
    #rd = get_portfolio_returns(ans4, rtn)
    
    plt.figure()
    ra.cumsum().plot(label='fast')
    rb.cumsum().plot(label='slow')
    #rc.cumsum().plot(label='fast new')
    #rd.cumsum().plot(label='slow new')
    
    plt.legend(loc='best', frameon=False)
    


def cache_momentum():
    for u in ['SMX', 'FTSE250']:
        cache_blind_momentum(u)


def main():
    cache_momentum()


if __name__ == '__main__':
    main()
