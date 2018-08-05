from quant.lib.main_utils import *
from quant.data import stocks
from quant.lib import portfolio_utils as pu, visualization_utils as vu
from matplotlib import pyplot as plt


# Simple lookback
def run_long(r, vol, pos_vol, i):
    rtn = r.divide(vol)
    rm = rtn.subtract(rtn.mean(axis=1), axis=0)
    slong = None
    tlong = -1
    l = np.arange(.3, 2.01, .1)
    s1 = rm.rolling(i).mean()
    s2 = rm.rolling(52, min_periods=13).mean().shift(i)
    for high in l:
        for low in l:
            p = 1. * ((s1 <= -low) & (s2 >= high))
            if p.sum(axis=1).sum() > 0:
                p = p[p > 0].ffill(limit=3).divide(pos_vol)
                g = p.abs().sum(axis=1)
                pnl = r.mul(p.shift()).sum(axis=1)[dt(2009,1,1):]
                pnl /= g.shift()[dt(2009,1,1):]
                pnl = pnl.fillna(0.)
                sr = pnl.mean() / pnl.std() * np.sqrt(52.)
                mu = pnl.mean() * 52.
                tot = pnl.sum()
                if sr > tlong:
                    pnl.name = '%d %.2f %.1f%% [%.1f %.1f]' % (i, sr, 100. * mu, high, -low)
                    tlong = sr
                    slong = pnl.cumsum().ffill()
    return slong
    

def run_short(r, vol, pos_vol, i):
    rtn = r.divide(vol)
    rm = rtn.subtract(rtn.mean(axis=1), axis=0)
    slong = None
    tlong = -1
    l = np.arange(.3, 2.01, .1)
    s1 = rm.rolling(i).mean()
    s2 = rm.rolling(52, min_periods=13).mean().shift(i)
    for high in l:
        for low in l:
            p = -1. * ((s1 >= low) & (s2 <= -high))
            if p.sum(axis=1).sum() < 0:
                p = p[p < 0].ffill(limit=3).divide(pos_vol)
                pnl = r.mul(p.shift()).sum(axis=1)[dt(2010,1,1):]
                sr = pnl.mean() / pnl.std() * np.sqrt(52.)
                tot = pnl.sum()
                if sr > tlong:
                    pnl.name = '%d %.2f [%.1f %.1f]' % (i, sr, -high, low)
                    tlong = sr
                    slong = pnl.cumsum()
    return slong


# Lookback, timing bottom
def run_long2(r, vol, pos_vol, i):
    rtn = r.divide(vol)
    rm = rtn.subtract(rtn.mean(axis=1), axis=0)
    slong = None
    tlong = -1
    l = np.arange(.3, 2.01, .1)
    s1 = rm.rolling(i).mean()
    s2 = rm.rolling(52, min_periods=13).mean().shift(i)
    acc = r.cumsum()
    ax = acc.rolling(i).min()
    for high in l:
        for low in l:
            p = 1. * ((s1 <= -low) & (s2 >= high) & (acc == ax))
            if p.sum(axis=1).sum() > 0:
                p = p[p > 0].ffill(limit=3).divide(pos_vol)
                g = p.abs().sum(axis=1)
                pnl = r.mul(p.shift()).sum(axis=1)[dt(2010,1,1):]
                pnl /= g.shift()[dt(2010,1,1):]
                pnl = pnl.fillna(0.)
                sr = pnl.mean() / pnl.std() * np.sqrt(52.)
                mu = pnl.mean() * 52.
                tot = pnl.sum()
                if tot > tlong:
                    pnl.name = '%d %.2f %.1f%% [%.1f %.1f]' % (i, sr, 100. * mu, high, -low)
                    tlong = tot
                    slong = pnl.cumsum().ffill()
    return slong


# Lookback, timing bottom, and timing top
def run_long3(r, vol, pos_vol, i):
    rtn = r.divide(vol)
    rm = rtn.subtract(rtn.mean(axis=1), axis=0)
    slong = None
    tlong = -1
    l = np.arange(.3, 2.01, .1)
    s1 = rm.rolling(i).mean()
    s2 = rm.rolling(52, min_periods=13).mean().shift(i)
    s3 = rtn.rolling(52, min_periods=13).mean().shift(i)
    acc = r.cumsum()
    ax = acc.rolling(i).min()
    for high in l:
        for low in l:
            p = 1. * ((s1 <= -low) & (s2 >= high) & (acc == ax) & (s3 >= 0.3))
            if p.sum(axis=1).sum() > 0:
                p = p[p > 0].ffill(limit=3).divide(pos_vol)
                g = p.abs().sum(axis=1)
                pnl = r.mul(p.shift()).sum(axis=1)[dt(2009,1,1):]
                pnl /= g.shift()[dt(2009,1,1):]
                pnl = pnl.fillna(0.)
                sr = pnl.mean() / pnl.std() * np.sqrt(52.)
                mu = pnl.mean() * 52.
                tot = pnl.sum()
                if tot > tlong:
                    pnl.name = '%d %.2f %.1f%% [%.1f %.1f]' % (i, sr, 100. * mu, high, -low)
                    tlong = tot
                    slong = pnl.cumsum().ffill()
    return slong


def estimate_reversal(universe='SMX'):
    r = stocks.load_google_returns(data_table=stocks.UK_STOCKS).resample('W').sum()
    u = stocks.get_universe(universe)
    r = r.loc[:, r.columns.isin(u.index)]
    w = r.abs()
    vol = w[w > 0].rolling(52, min_periods=13).median().ffill().bfill()
    vol2 = vol.copy()
    vol2[vol2 < .02] = .02
    vol3 = w[w > 0].rolling(13, min_periods=4).median().ffill().bfill()
    #a = pu.calc_drawdown_age(r)
    for i in xrange(1, 10):
        logger.info('Lookback %d' % i)
        pnl = run_long3(r, vol, vol2, i)
        if pnl is not None:
            pnl.plot()
    plt.legend(loc='best', frameon=False)

