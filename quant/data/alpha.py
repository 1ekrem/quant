from quant.lib.main_utils import *
from quant.lib import timeseries_utils as tu, data_utils as du
from quant.data import stocks
from statsmodels import api as sm

DATABASE_NAME = 'quant'
UK_ALPHA = 'uk_alpha'
UK_BETA_FACTORS = ['MCX', 'GBPUSD', 'EURGBP']


def create_alpha_table():
    du.create_t2_timeseries_table(DATABASE_NAME, UK_ALPHA)


def get_alpha(y, x, min_periods=63):
    my_y = y.dropna()
    my_x = sm.add_constant(x.loc[my_y.index].fillna(0.))
    if len(my_y) >= min_periods:
        lm = sm.OLS(my_y, my_x).fit()
        p = lm.pvalues
        p = p.loc[p.index != 'const']
        b = lm.params.loc[p.index]
        b[p > .05] *= 0.
    else:
        b = pd.Series([0.] * len(x.columns), index=x.columns)
    f = my_x.loc[:, b.index].mul(b, axis=1)
    s = my_y - f.sum(axis=1)
    return b, f, s
        


def estimate_alpha(y, x, stock, data_table):
    b, f, s = get_alpha(y, x)
    b.index = ['%s|%s' % (idx, stock) for idx in b.index]
    b = b.to_frame().T
    b.index = [y.index[-1]]
    tu.store_timeseries(b, DATABASE_NAME, data_table, 'Beta')
    s.name = 'Alpha|%s' % stock
    f.columns = ['%s|%s' % (idx, stock) for idx in f.columns]
    tu.store_timeseries(pd.concat([f, s], axis=1), DATABASE_NAME, data_table, 'Alpha')


def calculate_uk_alpha(lookback=5, latest=False, ids=None):
    u = stocks.get_ftse250_universe()
    u2 = stocks.get_ftse_smx_universe()
    u = pd.concat([u, u2.loc[~u2.index.isin(u.index)]], axis=0)
    x = stocks.load_google_returns(data_table=stocks.GLOBAL_ASSETS)
    x = x.loc[:, x.columns.isin(UK_BETA_FACTORS)]
    y = stocks.load_google_returns(data_table=stocks.UK_STOCKS)
    if ids is not None:
        y = y.loc[:, y.columns.isin(ids)]
    else:
        y = y.loc[:, y.columns.isin(u.index)]
    for idx in y.columns:
        my_y = y[idx].dropna()
        if not my_y.empty:
            ms = my_y.resample('AS').last()
            me = my_y.resample('A').last()
            first = len(ms) if latest else np.min([len(ms), lookback])
            for i in xrange(first -1, len(ms)):
                end = me.index[i]
                start = ms.index[np.max([0, i - lookback + 1])]
                yy = my_y[start:end].dropna()
                if len(yy) > 20:
                    logger.info('Calculating %s %d-%d' % (idx, start.year, end.year))
                    estimate_alpha(yy, x[start:end], idx, UK_ALPHA)


def load_alpha(start_date=None, end_date=None, data_table=UK_ALPHA, data_name='Alpha'):
    data = stocks.load_google_returns(start_date, end_date, data_name, data_table)
    data.columns = pd.MultiIndex.from_tuples([tuple(x.split('|')) for x in data.columns], names=['Factor', 'Stock'])
    return data
        
    