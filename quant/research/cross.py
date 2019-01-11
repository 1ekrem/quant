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
    #rtn2 = rc.resample('W').sum().apply(_get_first, axis=0)
    w = rtn.abs()
    vol = w[w > 0].rolling(52, min_periods=13).median()
    vol[vol < 5e-3] = 5e-3
    vol2 = vol.copy()
    vol2[vol2 < .02] = .02
    #rv = rtn2.divide(vol)
    rv = rtn.divide(vol)
    rm = rv.subtract(rv.median(axis=1), axis=0)
    return rtn, rm, vol2


def get_volume(vol):
    v = vol.resample('W').sum()
    v = v.rolling(52, min_periods=13).median()
    v[v <= 0.] = .5
    ans = np.log(v)
    ans = ans.subtract(ans.mean(axis=1), axis=0).divide(ans.std(axis=1), axis=0)
    return ans


def get_spread(s):
    return s.iloc[-20:].mean(axis=0)


def get_universe_returns(universe, data_name='Returns'):
    u = stocks.load_universe(universe)
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS, data_name=data_name, tickers=u.index)
    return r.reindex(u.index, axis=1)


def get_dataset(universe, max_rtn=None, max_spread=None):
    r = get_universe_returns(universe)
    volume = get_universe_returns(universe, data_name='Volume')
    if max_spread is not None:
        spread = get_universe_returns(universe, data_name='Spread')
        spread = get_spread(spread)
        spread = spread[spread <= max_spread]
        r = r.loc[:, r.columns.isin(spread.index)]
        volume = volume.loc[:, volume.columns.isin(spread.index)]
    if max_rtn is None:
        if universe in ['SMX', 'AIM']:
            max_rtn = .25
        else:
            max_rtn = .2
    r, rm, v = get_returns(r, max_rtn=max_rtn)
    vol = get_volume(volume)
    return r, rm, v, vol


def load_fundamental_changes(data_type, u):
    data = stocks.load_financial_data(data_type, u.index)
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
    return _get_first(ra.fillna(0.))



class Momentum(object):
    def __init__(self, universe, max_spread=.02, min_stocks=3.):
        self.universe = universe
        self.max_spread = max_spread
        self.min_stocks = min_stocks
        self.run()
        
    def run(self):
        logger.info('Running momentum on %s' % self.universe)
        self.load_dataset()
        self.run_all()
    
    def load_dataset(self):
        logger.info('Loading returns')
        self.rtn, self.rm, self.vol, self.volume = get_dataset(self.universe, max_spread=self.max_spread)
        acc2 = self.rtn.cumsum()
        self.dd = acc2.rolling(13, min_periods=1).max() - acc2
        logger.info('Loading fundamental data')
        self.financials = load_financials(self.universe)
        data = get_financials_overall_score(self.financials)
        self.score = tu.resample(data, self.rtn).reindex(self.rtn.columns, axis=1)
    
    def run_sim(self, stm=3, ns=10, min_fast=0., min_slow=0., fast=True, fundamental=False, good=False):
        s1 = -1. * get_stock_mom(self.rm, stm)
        s2 = get_stock_mom(self.rm, 52).shift(stm)
        s3 = get_stock_mom(self.rm, 52)
        holding = 0
        input1 = s1 if fast else s1 + s2
        if fundamental:
            input2 = self.score
        else:
            input2 = s1 + s2 if fast else s1
        f = 1. * (s1 >= min_fast) * (s3 >= min_slow)
        if good:
            g = s3.subtract(s3.median(axis=1), axis=0)
            f = f * (g >= 0)
        ans = get_step_positions(input1, input2, self.vol, ns, f, None, holding=holding)
        ra = get_portfolio_returns(ans, self.rtn)
        return ans, ra
    
    def get_sim_analytics(self, ans, ra):
        s = pd.Series([])
        end_date = dt(2018, 10, 31)
        s.loc['total'] = ra[:end_date].sum()
        s.loc['recent'] = ra[dt(2014,1,1):end_date].sum()
        s.loc['n'] = (ans > 0).sum(axis=1).mean()
        s.loc['nc'] = (ans > 0)[dt(2014, 1, 1):].sum(axis=1).mean()
        return s
    
    def decide(self, analytics):
        a = analytics.iloc[:, :2].mul([.4,.6], axis=1).mean(axis=1)
        n = analytics.iloc[:, 2:].mean(axis=1)
        c = a[n >= self.min_stocks]
        c = c[c == c.max()]
        return c.index[0]
        
    def run_pass(self, sets):
        analytics = []
        for idx, msg, s in sets:
            logger.info(msg)
            ans, ra = self.run_sim(**s)
            a = self.get_sim_analytics(ans, ra)
            a.name = idx
            analytics.append(a)
        analytics = pd.concat(analytics, axis=1).T
        return analytics
        
    def get_stm_set(self, base):
        ans = []
        x = np.arange(2, 14)
        for i, k in enumerate(x):
            msg = 'stm: %d .. %.1f%%' % (k, 100. * (i + 1) / len(x))
            new = base.copy()
            new['stm'] = k
            ans.append((k, msg, new))
        return ans
    
    def get_stocks_set(self, base):
        ans = []
        x = np.arange(3, 21)
        for i, k in enumerate(x):
            msg = 'n: %d .. %.1f%%' % (k, 100. * (i + 1) / len(x))
            new = base.copy()
            new['ns'] = k
            ans.append((k, msg, new))
        return ans
    
    def get_fast_set(self, base):
        ans = []
        x = np.arange(0., 1.51, .1)
        for i, k in enumerate(x):
            msg = 'fast: %.1f .. %.1f%%' % (k, 100. * (i + 1) / len(x))
            new = base.copy()
            new['min_fast'] = k
            ans.append((k, msg, new))
        return ans

    def get_slow_set(self, base):
        ans = []
        x = np.arange(0., 1.01, .1)
        for i, k in enumerate(x):
            msg = 'slow: %.1f .. %.1f%%' % (k, 100. * (i + 1) / len(x))
            new = base.copy()
            new['min_slow'] = k
            ans.append((k, msg, new))
        return ans

    def get_drawdown_set(self, base):
        ans = []
        x = np.arange(0., .16, .01)
        for i, k in enumerate(x):
            msg = 'drawdown: %.2f .. %.1f%%' % (k, 100. * (i + 1) / len(x))
            new = base.copy()
            new['min_dd'] = k
            ans.append((k, msg, new))
        return ans

    def get_fast(self):
        return {'fast': True, 'fundamental': False}
    
    def get_slow(self):
        return {'fast': False, 'fundamental': False}
    
    def get_fast_fundamental(self):
        return {'fast': True, 'fundamental': True}
    
    def get_slow_fundamental(self):
        return {'fast': False, 'fundamental': True}
    
    def get_good(self):
        return {'fast': True, 'fundamental': False, 'good': True}

    def run_test(self, base):

        logger.info('Testing fast')
        set2 = self.get_fast_set(base)
        a2 = self.run_pass(set2)
        min_fast = self.decide(a2)
        base['min_fast'] = min_fast

        logger.info('Testing slow')
        set4 = self.get_slow_set(base)
        a4 = self.run_pass(set4)
        min_slow = self.decide(a4)
        base['min_slow'] = min_slow

        logger.info('Testing STM')
        set = self.get_stm_set(base)
        a = self.run_pass(set)
        stm = self.decide(a)
        base['stm'] = stm

        logger.info('Testing ns')
        set3 = self.get_stocks_set(base)
        a3 = self.run_pass(set3)
        ns = self.decide(a3)
        base['ns'] = ns

        return base, a3.loc[ns]

    def run_fast(self):
        base = self.get_fast()
        return self.run_test(base)
        
    def run_slow(self):
        base = self.get_slow()
        return self.run_test(base)
    
    def run_fast_fundamental(self):
        base = self.get_fast_fundamental()
        return self.run_test(base)
        
    def run_slow_fundamental(self):
        base = self.get_slow_fundamental()
        return self.run_test(base)
    
    def run_good(self):
        base = self.get_good()
        return self.run_test(base)

    def run_all(self):
        ans = {}
        ans['Fast'] = self.run_fast()
        ans['Slow'] = self.run_slow()
        ans['Fast fundamental'] = self.run_fast_fundamental()
        ans['Slow fundamental'] = self.run_slow_fundamental()
        ans['Good'] = self.run_good()
        self.results = ans
    
    def patch_results(self):
        self.results = {'Fast': [{'fast': True, 'fundamental': False,
                                              'min_fast': 0., 'min_slow': 0.,
                                              'ns': 4, 'stm': 3}],
                        'Fast Fundamental': [{'fast': True, 'fundamental': True,
                                              'min_fast': 0., 'min_slow': 0.,
                                              'ns': 4, 'stm': 3}],
                        'Slow': [{'fast': False, 'fundamental': False,
                                              'min_fast': 0., 'min_slow': 0.,
                                              'ns': 4, 'stm': 3}],
                        'Slow Fundamental': [{'fast': False, 'fundamental': True,
                                              'min_fast': 0., 'min_slow': 0.,
                                              'ns': 4, 'stm': 3}]}

    def plot_simulations(self):
        plt.figure()
        for k, v in self.results.iteritems():
            ans, r = self.run_sim(**v[0])
            r.cumsum().plot(label='%s %.2f' % (k, r.sum()))
        plt.legend(loc='best', frameon=False)
        plt.title(self.universe, weight='bold')

    def get_positions(self):
        pos = {}
        for k, v in self.results.iteritems():
            ans, r = self.run_sim(**v[0])
            pos[k] = ans
        return pos
