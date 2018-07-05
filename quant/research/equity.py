from quant.lib.main_utils import *
from quant.data import stocks
from quant.lib import timeseries_utils as tu, portfolio_utils as pu, \
    machine_learning_utils as mu

STOCK_VOL_FLOOR = 0.02


def calculate_signal_positions(signal, top=5, long_only=True):
    def calc(x):
        ans = x * np.nan
        tmp = x.dropna()
        if not tmp.empty:
            tmp = tmp.sort_values()
            if len(tmp) > top:
                tmp = tmp.iloc[-top:]
            ans.loc[tmp.index] = 1.
        return ans
    
    return signal.apply(calc, axis=1)


# Simulations
class MomentumSim(object):
    '''
    Stocks strategy
    '''
    def __init__(self, start_date, end_date, sample_date, universe, simulation_name, max_depth=5, model_path=MODEL_PATH, 
                 load_model=False, cross_validation_buskcets=10, top=5, holding_period=4, long_only=True):
        self.simulation_name = simulation_name
        self.start_date = start_date
        self.end_date = end_date
        self.sample_date = sample_date
        self.universe = universe
        self.max_depth = max_depth
        self.optimal_depth = max_depth
        self.model_path = model_path
        self.load_model = load_model
        self.cross_validation_buckets = cross_validation_buskcets
        self.top = top
        self.long_only = long_only
        self.holding_period = holding_period
        self.run_sim()
    
    def run_sim(self):
        logger.info('Running simulation %s' % self.simulation_name)
        self.model = None
        self.load_universe()
        self.load_stock_data()
        if self.load_model:
            self.load_existing_model(self._r)
        if self.model is None:
            self.find_optimal_depth()
            self.build_model()
        self.calculate_signals()
        self.calculate_returns()

    def load_universe(self):
        logger.info('Loading universe')
        self.u = stocks.get_universe(self.universe)

    def load_stock_data(self):
        logger.info('Loading stock returns')
        r = stocks.load_google_stock_returns(self.start_date - relativedelta(years=1), self.end_date)
        self.stock_returns = r.loc[:, r.columns.isin(self.u.index)]
        self.asset_names = self.stock_returns.columns
        logger.info('Calculating volatility')
        w = self.stock_returns.resample('W').sum().abs()
        v = w[w > 0].rolling(52, min_periods=13).median().ffill().bfill().fillna(0.)
        v[v < STOCK_VOL_FLOOR] = STOCK_VOL_FLOOR
        self.stock_vol = v
        self._r = self.stock_returns.cumsum().resample('W').last().diff()
        self.v = tu.resample(v, self._r).ffill().bfill()
        self.r = self._r.divide(self.v)
        
    def create_estimation_data(self, depth):
        lookbacks = [3 ** (i+1) for i in xrange(depth)]
        ans = dict([('S%d' % i, self.r.ewm(span=i).mean().shift(2)) for i in lookbacks])
        ans['Rev'] = self.r.rolling(2, min_periods=1).mean()
        ans['Vol'] = self.stock_vol
        return ans

    def estimate_model(self, x, timeline, asset_returns=None, model=None):
        return mu.StockRandomBoostingComponent(x, timeline, self.asset_names, asset_returns=asset_returns,
                                               model=model, cross_validation_buckets=self.cross_validation_buckets)

    def build_model(self):
        y = mu.get_score(self.r, 0, 1.5).shift(-1)[self.start_date:self.end_date]
        x = self.create_estimation_data(self.optimal_depth)
        self._model = self.estimate_model(x, y, asset_returns=y)
        self.model = self._model.model

    def find_optimal_depth(self):
        y = self.r[self.start_date:self.end_date]
        error_rates = pd.Series([])
        for depth in xrange(1, self.max_depth + 1):
            logger.info('Testing depth %d' % depth)
            x = self.create_estimation_data(depth)
            m = self.estimate_model(x, y, asset_returns=y)
            m.run_cross_validation()
            error_rates.loc[depth] = m.error_rate
        self.error_rates = error_rates
        logger.info('Error rates:\n%s' % str(error_rates))
        self.optimal_depth = self.error_rates.index[self.error_rates == self.error_rates.min()][0]
        logger.info('Optimal depth: %d' % self.optimal_depth)

    def calculate_signals(self):
        logger.info('Calculating signals')
        x = self.create_estimation_data(self.optimal_depth)
        y = self.r[self.start_date:self.end_date]
        m = self.estimate_model(x, y, model=self.model)
        self.signals = m.signals

    def calculate_returns(self):
        if self.signals is not None:
            logger.info('Simulating portfolio')
            self._pos = calculate_signal_positions(self.signals[~self._r.isnull()], self.top, self.long_only)
            self.positions = tu.resample(self._pos.ffill(limit=self.holding_period).divide(self.stock_vol), self.stock_returns)
            self.pnl = self.stock_returns.mul(self.positions).sum(axis=1)[self.start_date:]
            self.analytics = pu.get_returns_analytics(self.pnl.to_frame())
    
    def get_model_filename(self):
        return '%s%s.model' % (self.model_path, self.simulation_name)

    def load_existing_model(self, timeline):
        logger.info('Loading model %s' % self.simulation_name)
        filename = self.get_model_filename()
        load_data = load_pickle(filename)
        if load_data is not None:
            self.optimal_depth, self.error_rate, model = load_data

    def pickle_model(self):
        filename = self.get_model_filename()
        if self.model is not None:
            logger.info('Exporting model')
            data = self.optimal_depth, self.error_rate, self.model
            write_pickle(data, filename)

    