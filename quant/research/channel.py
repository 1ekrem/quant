from quant.lib.main_utils import *
from quant.data import stocks
from quant.research import cross
from quant.lib import timeseries_utils as tu
from scipy import optimize as op
from matplotlib import pyplot as plt

PATH = '/home/wayne/TempWork/channel/'
make_dir(PATH)


def get_dataset(universe, max_rtn=None, max_spread=None):
    r = cross.get_universe_returns(universe)
    if max_spread is not None:
        spread = cross.get_universe_returns(universe, data_name='Spread')
        spread = cross.get_spread(spread)
        spread = spread[spread <= max_spread]
        r = r.loc[:, r.columns.isin(spread.index)]
    if max_rtn is None:
        if universe in ['SMX', 'AIM']:
            max_rtn = .25
        else:
            max_rtn = .2
    rs, rm, v = cross.get_returns(r, max_rtn=max_rtn)
    return r, rs, rm, v


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


def get_pivot_index(x_in, low_touch, dlow, dmin):
    x = x_in[low_touch]
    if len(x) > 1:
        x = x[np.hstack((np.array([True]), dlow > dmin))]
    return x[-1]


def get_channel_stats(x_in, y_in, y, b, h, margin, head_period, gap_period):
    ans = {}
    dmin = np.max((len(x_in) * gap_period, 3.))
    hmin = np.int(np.max((len(x_in) * head_period, 3.)))
    low = b + margin * h
    high = b + (1. - margin) * h
    low_touch = y_in < low
    high_touch = y_in > high
    sig = np.zeros(len(y_in)) * np.nan
    sig[low_touch] = 1.
    sig[high_touch] = -1.
    sig = pd.Series(sig).ffill().values[-1]
    if y[-1] > b + h:
        sig = 2.
    elif y[-1] < b:
        sig = -2.
    z = y[-1] / h
    dlow = np.diff(x_in[low_touch])
    dhigh = np.diff(x_in[high_touch])
    f = np.sum(low_touch) + np.sum(high_touch) - np.sum(dhigh <= dmin) - np.sum(dlow <= dmin)
    plow = get_pivot_index(x_in, low_touch, dlow, dmin)
    phigh = get_pivot_index(x_in, high_touch, dhigh, dmin)    
    ans['head'] = np.any(high_touch[:hmin]) or np.any(low_touch[:hmin])
    ans['tail'] = np.any(high_touch[-hmin:]) or np.any(low_touch[-hmin:])
    ans['signal'] = sig
    ans['z'] = z
    ans['points'] = f
    ans['pivot'] = np.min((plow, phigh))
    return ans

    
def calculate_channel(r, margin=.025, shift_period=2, head_period=.05, gap_period=.05):
    p = trim(r).cumsum().values
    x = get_x(p)
    p_in = p[:-shift_period]
    x_in = x[:-shift_period]
    o = op.fmin_slsqp(get_width, 0., args=(p_in, x_in), iprint=0)
    k = o[0]
    y = get_y(p, x, k)
    y_in = y[:-shift_period]
    b = np.min(y_in)
    h = np.max(y_in) - b
    ans = get_channel_stats(x_in, y_in, y, b, h, margin, head_period, gap_period)
    return k, b, h, ans


def search_channel(r, margin=.025, shift_period=2, gap_period=.05):
    k, b, h, ans = calculate_channel(r, margin=margin, shift_period=shift_period, gap_period=gap_period)
    rs = trim(r).iloc[ans.get('pivot'):]
    return k, b, h, ans, rs


def get_channel_set(r, margin=.025, shift_period=2, gap_period=.05, min_lookback=40, diagnose=False):
    results = []
    _, _, _, _, rs = search_channel(r, margin=margin, shift_period=shift_period, gap_period=gap_period)
    i = 0
    while len(rs) > min_lookback and i < 20:
        k, b, h, ans, rx = search_channel(rs, margin=margin, shift_period=shift_period, gap_period=gap_period)
        if diagnose:
            plt.figure()
            rs.cumsum().plot()
            plot_channel(rs, k, b, h)
            plt.title('%d %d' % (len(rs), ans.get('points')))
        results.append((k, b, h, ans, rs))
        rs = rx
        i += 1
    return results


def get_signal(r):
    rtn = trim(r)
    ans = np.nan * rtn.resample('W').last()
    ans2 = ans.copy()
    for idx in ans.index[52:]:
        a = get_channel_set(rtn[:idx])
        if len(a) > 0:
            ans.loc[idx] = a[-1][3].get('signal')
            ans2.loc[idx] = a[-1][3].get('z')
    return ans, ans2


def cache_channel_signal(universe='SMX'):
    rtn, rs, rm, v = get_dataset(universe)
    score = []
    points = []
    for c in rtn.columns:
        logger.info('Running %s' % c)
        s, p = get_signal(rtn.loc[:, c])
        score.append(s)
        points.append(p)
    score = pd.concat(score, axis=1)
    points = pd.concat(points, axis=1)
    write_pickle((score, points), PATH + universe + ' signal.dat')


def load_channel_signal(universe='SMX'):
    return load_pickle(PATH + universe + ' signal.dat')


def test_combo(universe='SMX'):
    rtn, rs, rm, vol = get_dataset(universe, max_spread=.02)
    s, p, s2, p2 = load_channel_signal(universe)
    #f = cross.load_financials(universe)
    #s = tu.resample(cross.get_financials_overall_score(f), rtn)
    
    plt.figure()
    pos = (s == 2) * (s2 == -1).divide(vol)
    cross.get_portfolio_returns(pos.ffill(limit=3), rs).cumsum().plot(label='Revert break')
    pos2 = (s == 2) * (s2 == 1).divide(vol)
    cross.get_portfolio_returns(pos2.ffill(limit=3), rs).cumsum().plot(label='Trend break')
    pos3 = (s == 2) * (s2 == 2).divide(vol)
    cross.get_portfolio_returns(pos2.ffill(limit=3), rs).cumsum().plot(label='Both break')
    plt.legend(loc='best', frameon=False)
    plt.title('Break', weight='bold')
    
    plt.figure()
    pos = (s == 1) * (s2 == -1).divide(vol)
    cross.get_portfolio_returns(pos.ffill(limit=3), rs).cumsum().plot(label='Revert')
    pos2 = (s == 1) * (s2 == 1).divide(vol)
    cross.get_portfolio_returns(pos2.ffill(limit=3), rs).cumsum().plot(label='Trend')
    pos3 = (s == 1) * (s2 == 2).divide(vol)
    cross.get_portfolio_returns(pos2.ffill(limit=3), rs).cumsum().plot(label='Follow break')
    plt.legend(loc='best', frameon=False)
    plt.title('Follow', weight='bold')

    plt.figure()
    pos = (s == -2) * (s2 == -1).divide(vol)
    cross.get_portfolio_returns(pos.ffill(limit=3), rs).cumsum().plot(label='break down')
    pos2 = (s == -2) * (s2 == 1).divide(vol)
    cross.get_portfolio_returns(pos2.ffill(limit=3), rs).cumsum().plot(label='break up')
    pos3 = (s == -2) * (s2 == 2).divide(vol)
    cross.get_portfolio_returns(pos2.ffill(limit=3), rs).cumsum().plot(label='break')
    plt.legend(loc='best', frameon=False)
    plt.title('Break', weight='bold')
    
    plt.figure()
    pos = (s == -1) * (s2 == -1).divide(vol)
    cross.get_portfolio_returns(pos.ffill(limit=3), rs).cumsum().plot(label='drop down')
    pos2 = (s == -1) * (s2 == 1).divide(vol)
    cross.get_portfolio_returns(pos2.ffill(limit=3), rs).cumsum().plot(label='drop up')
    pos3 = (s == -1) * (s2 == 2).divide(vol)
    cross.get_portfolio_returns(pos2.ffill(limit=3), rs).cumsum().plot(label='drop break')
    plt.legend(loc='best', frameon=False)
    plt.title('Follow', weight='bold')

    
def plot_channel(r, k, b, h):
    rtn = trim(r)
    x = get_x(rtn)
    low = pd.Series(k * x + b, index=rtn.index, name='Low')
    high = pd.Series(k * x + b + h, index=rtn.index, name='High')
    low.plot(ls='--', color='green')
    high.plot(ls='--', color='green')


class Momentum(object):
    def __init__(self, universe, max_spread=.02, min_stocks=3., channel_type='All'):
        self.universe = universe
        self.max_spread = max_spread
        self.min_stocks = min_stocks
        self.channel_type = channel_type
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
        s, p = load_channel_signal(self.universe)
        self.sig = s.reindex(self.rtn.columns, axis=1).fillna(0.)
        self.z = p.reindex(self.rtn.columns, axis=1).fillna(0.)

    def run_sim(self, stm=3, ns=15, min_fast=0., min_slow=0., max_z=1., fast=True, fundamental=False):
        s1 = -1. * cross.get_stock_mom(self.rm, stm)
        s2 = cross.get_stock_mom(self.rm, 52).shift(stm)
        s3 = cross.get_stock_mom(self.rm, 52)
        holding = 0
        input1 = s1 if fast else s2
        if fundamental:
            input2 = self.score
        else:
            input2 = s2 if fast else s1
        f = 1. * (s1 >= min_fast) * (s3 >= min_slow) * (self.z <= max_z) * (self.z >= 0)
        ans = cross.get_step_positions(input1, input2, self.vol, ns, f, None, holding=holding)
        ra = cross.get_portfolio_returns(ans, self.rtn)
        return ans, ra
    
    def get_sim_analytics(self, ans, ra):
        end_date = dt(2018, 10, 31)
        s = pd.Series([])
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
        x = np.arange(0., 1.1, .1)
        for i, k in enumerate(x):
            msg = 'fast: %.1f .. %.1f%%' % (k, 100. * (i + 1) / len(x))
            new = base.copy()
            new['min_fast'] = k
            ans.append((k, msg, new))
        return ans

    def get_slow_set(self, base):
        ans = []
        x = np.arange(0., 1.1, .1)
        for i, k in enumerate(x):
            msg = 'slow: %.1f .. %.1f%%' % (k, 100. * (i + 1) / len(x))
            new = base.copy()
            new['min_slow'] = k
            ans.append((k, msg, new))
        return ans

    def get_z_set(self, base):
        ans = []
        x = np.arange(0.1, 1.01, .1)
        for i, k in enumerate(x):
            msg = 'z: %.1f .. %.1f%%' % (k, 100. * (i + 1) / len(x))
            new = base.copy()
            new['max_z'] = k
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
        
        logger.info('Testing z')
        set2 = self.get_z_set(base)
        a2 = self.run_pass(set2)
        max_z = self.decide(a2)
        base['max_z'] = max_z

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
        ans['Fast f'] = self.run_fast_fundamental()
        ans['Slow f'] = self.run_slow_fundamental()
        self.results = ans
    
    def patch_results(self):
        self.results = {'Fast': [{'fast': True, 'fundamental': False,
                                              'min_fast': 2, 'min_slow': 0,
                                              'ns': 5}],
                        'Fast Fundamental': [{'fast': True, 'fundamental': True,
                                              'min_fast': 2, 'min_slow': 0,
                                              'ns': 4}],
                        'Slow': [{'fast': False, 'fundamental': False,
                                              'min_fast': 2, 'min_slow': 0,
                                              'ns': 4}],
                        'Slow Fundamental': [{'fast': False, 'fundamental': True,
                                              'min_fast': 2, 'min_slow': 0,
                                              'ns': 4}]}

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


def main():
    cache_channel_signal('SMX')


if __name__ == '__main__':
    main()


    