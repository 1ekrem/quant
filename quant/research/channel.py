from quant.lib.main_utils import *
from quant.data import stocks
from quant.research import cross
from quant.lib import timeseries_utils as tu
from scipy import optimize as op
from matplotlib import pyplot as plt

PATH = '/home/wayne/TempWork/channel/'
make_dir(PATH)


def get_ztf(lookback=13):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS, data_name='Returns', tickers=['ZTF'])
    return r.resample('W').sum().iloc[-lookback:, 0]


def trim(r):
    x = r[~r.isnull()]
    return r.loc[x.index[0]:x.index[-1]].fillna(0.)


def get_x(r):
    return np.arange(len(r))


def get_y(p, x, k):
    return p - x * k


def get_width(k, p, x):
    y = get_y(p, x, k)
    return np.max(y) - np.min(y)


def find_channel(r, margin=.025, touch_lookback=3):
    p = trim(r).cumsum().values
    x = get_x(p)
    o = op.fmin_slsqp(get_width, 0., args=(p, x), iprint=0)
    k = o[0]
    y = get_y(p, x, k)
    b = np.min(y)
    h = np.max(y) - b
    x1 = b + margin * h
    x2 = b + (1. - margin) * h
    low_touch = y < x1
    high_touch = y > x2
    f = np.sum(low_touch) + np.sum(high_touch)
    th = np.any(high_touch[-touch_lookback:])
    tl = np.any(low_touch[-touch_lookback:])
    return k, b, h, f, th, tl


def get_breakthrough_signal(r, lookback=20, margin=.025, touch_lookback=3):
    rtn = trim(r)
    ans = pd.Series([])
    for i in xrange(lookback, len(rtn)):
        s = rtn.iloc[i - lookback:i]
        p = s.cumsum()
        k, b, h, f, th, tl = find_channel(s.iloc[:-1], margin, touch_lookback)
        x = get_x(s)
        upper = k * x + b + h
        lower = k * x + b
        signal = np.nan
        if f > 3:
            if p.iloc[-1] < lower[-1] and p.iloc[-2] >= lower[-2]:
                signal = -1
            elif p.iloc[-1] > upper[-1] and p.iloc[-2] <= upper[-2]:
                signal = 1
        ans.loc[s.index[-1]] = signal
    return ans


def get_support_signal(r, lookback=20, margin=.025, touch_lookback=3):
    rtn = trim(r)
    ans = pd.Series([])
    for i in xrange(lookback, len(rtn)):
        s = rtn.iloc[i - lookback:i]
        p = s.cumsum()
        k, b, h, f, th, tl = find_channel(s.iloc[:-1], margin, touch_lookback)
        x = get_x(s)
        upper = k * x + b + h
        lower = k * x + b
        signal = np.nan
        if f > 3:
            if p.iloc[-1] > lower[-1] and tl:
                signal = 1
            elif p.iloc[-1] < upper[-1] and th:
                signal = -1
        ans.loc[s.index[-1]] = signal
    return ans


def find_breakthrough(r, v):
    logger.info('Running %s' % r.name)
    t1 = -100.
    s1 = None
    t2 = -100.
    s2 = None
    t3 = -100.
    s3 = None
    t4 = -100.
    s4 = None
    for s in [8, 13, 20, 26, 52, 78, 104]:
        sig = get_breakthrough_signal(r, s)
        pos = sig.divide(v)
        rtn = r.mul(pos.ffill(limit=1).shift()).sum()
        rtn2 = r.mul(pos.ffill(limit=2).shift()).sum()
        rtn3 = r.mul(pos.ffill(limit=3).shift()).sum()
        rtn4 = r.mul(pos.ffill(limit=7).shift()).sum()
        if rtn > t1:
            t1 = rtn
            s1 = sig.ffill(limit=1)
        if rtn2 > t2:
            t2 = rtn2
            s2 = sig.ffill(limit=2)
        if rtn3 > t3:
            t3 = rtn3
            s3 = sig.ffill(limit=3)
        if rtn4 > t4:
            t4 = rtn4
            s4 = sig.ffill(limit=7)
    ans = pd.concat([s1, s2, s3, s4], axis=1).sum(axis=1)
    ans.name = r.name
    return ans


def find_support(r, v):
    logger.info('Running %s' % r.name)
    t1 = -100.
    s1 = None
    t2 = -100.
    s2 = None
    t3 = -100.
    s3 = None
    t4 = -100.
    s4 = None
    for s in [8, 13, 20, 26, 52, 78, 104]:
        sig = get_support_signal(r, s)
        pos = sig.divide(v)
        rtn = r.mul(pos.ffill(limit=1).shift()).sum()
        rtn2 = r.mul(pos.ffill(limit=2).shift()).sum()
        rtn3 = r.mul(pos.ffill(limit=3).shift()).sum()
        rtn4 = r.mul(pos.ffill(limit=7).shift()).sum()
        if rtn > t1:
            t1 = rtn
            s1 = sig.ffill(limit=1)
        if rtn2 > t2:
            t2 = rtn2
            s2 = sig.ffill(limit=2)
        if rtn3 > t3:
            t3 = rtn3
            s3 = sig.ffill(limit=3)
        if rtn4 > t4:
            t4 = rtn4
            s4 = sig.ffill(limit=7)
    ans = pd.concat([s1, s2, s3, s4], axis=1).sum(axis=1)
    ans.name = r.name
    return ans

    
def cache_breakthrough_score(universe='SMX'):
    rtn, rm, vol, volume = cross.get_dataset(universe, max_spread=None)
    score = pd.concat([find_breakthrough(rtn.loc[:, c], vol.loc[:, c]) for c in rtn.columns], axis=1)
    write_pickle(score, PATH + universe + '_break.dat')


def cache_support_score(universe='SMX'):
    rtn, rm, vol, volume = cross.get_dataset(universe, max_spread=None)
    score = pd.concat([find_support(rtn.loc[:, c], vol.loc[:, c]) for c in rtn.columns], axis=1)
    write_pickle(score, PATH + universe + '_sup.dat')


def load_breakthrough_score(universe='SMX'):
    return load_pickle(PATH + universe + '_break.dat')


def load_support_score(universe='SMX'):
    return load_pickle(PATH + universe + '_sup.dat')


def test_combo(universe='SMX'):
    rtn, rm, vol, volume = cross.get_dataset(universe, max_spread=.02)
    f = cross.load_financials(universe)
    s = tu.resample(cross.get_financials_overall_score(f), rtn)
    b = load_breakthrough_score(universe).reindex(rtn.columns, axis=1)
    b2 = load_support_score(universe).reindex(rtn.columns, axis=1)
    pos = (b > 0).divide(vol)
    pos2 = (b2 > 0).divide(vol)
    pos3 = (b2 > 0) * (b > 0) * (s >= 0).divide(vol)
    cross.get_portfolio_returns(pos, rtn).cumsum().plot(label='Breakthrough')
    cross.get_portfolio_returns(pos2, rtn).cumsum().plot(label='Support')
    cross.get_portfolio_returns(pos3, rtn).cumsum().plot(label='Combo')
    plt.legend(loc='best', frameon=False)
    

def plot_channel(r, k, b, h):
    rtn = trim(r)
    x = get_x(rtn)
    low = pd.Series(k * x + b, index=rtn.index, name='Low')
    high = pd.Series(k * x + b + h, index=rtn.index, name='High')
    low.plot(ls='--', color='green')
    high.plot(ls='--', color='green')


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
        self.rtn, self.rm, self.vol, self.volume = cross.get_dataset(self.universe, max_spread=self.max_spread)
        logger.info('Loading fundamental data')
        self.financials = cross.load_financials(self.universe)
        x = cross.get_financials_overall_score(self.financials)
        self.score = tu.resample(x, self.rtn).reindex(self.rtn.columns, axis=1)
        self.s1 = load_breakthrough_score(self.universe).reindex(self.rtn.columns, axis=1)
        self.s2 = load_support_score(self.universe).reindex(self.rtn.columns, axis=1)

    def run_sim(self, stm=3, ns=10, min_fast=0., min_slow=0., fast=True, fundamental=False):
        s1 = self.s1
        s2 = self.s2
        s3 = cross.get_stock_mom(self.rm, stm)
        holding = 0
        input1 = s1 if fast else s2
        if fundamental:
            input2 = self.score
        else:
            input2 = s2 if fast else s1
        f = 1. * (s1 >= min_fast) * (s2 >= min_slow) * (self.score >= 0) * (s1 >= 0) * (s2 >= 0)
        ans = cross.get_step_positions(input1, input2, self.vol, ns, f, None, holding=holding)
        ra = cross.get_portfolio_returns(ans, self.rtn)
        return ans, ra
    
    def get_sim_analytics(self, ans, ra):
        s = pd.Series([])
        s.loc['total'] = ra.sum()
        s.loc['recent'] = ra[dt(2014,1,1):].sum()
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
        x = np.arange(3, 14)
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
        x = np.arange(-1, 2.1)
        for i, k in enumerate(x):
            msg = 'fast: %.1f .. %.1f%%' % (k, 100. * (i + 1) / len(x))
            new = base.copy()
            new['min_fast'] = k
            ans.append((k, msg, new))
        return ans

    def get_slow_set(self, base):
        ans = []
        x = np.arange(-1, 2.1)
        for i, k in enumerate(x):
            msg = 'slow: %.1f .. %.1f%%' % (k, 100. * (i + 1) / len(x))
            new = base.copy()
            new['min_slow'] = k
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
    
    def run_all(self):
        ans = {}
        ans['Fast'] = self.run_fast()
        ans['Slow'] = self.run_slow()
        ans['Fast fundamental'] = self.run_fast_fundamental()
        ans['Slow fundamental'] = self.run_slow_fundamental()
        self.results = ans
    
    def patch_results(self):
        self.results = {'Fast': [{'fast': True, 'fundamental': False,
                                              'min_fast': 1.5, 'min_slow': 0.,
                                              'ns': 5, 'stm': 3}],
                        'Fast Fundamental': [{'fast': True, 'fundamental': True,
                                              'min_fast': 1.4, 'min_slow': -.1,
                                              'ns': 4, 'stm': 3}],
                        'Slow': [{'fast': False, 'fundamental': False,
                                              'min_fast': 1.4, 'min_slow': -.1,
                                              'ns': 4, 'stm': 3}],
                        'Slow Fundamental': [{'fast': False, 'fundamental': True,
                                              'min_fast': 1.4, 'min_slow': -.1,
                                              'ns': 4, 'stm': 3}]}

    def plot_simulations(self):
        plt.figure()
        for k, v in self.results.iteritems():
            ans, r = self.run_sim(**v[0])
            r.cumsum().plot(label='%s %.2f' % (k, r.sum()))
        plt.legend(loc='best', frameon=False)
        plt.title(self.universe, weight='bold')


def main():
    cache_breakthrough_score()
    cache_support_score()


if __name__ == '__main__':
    main()


    