from quant.lib.main_utils import *
from quant.data import stocks
from quant.lib import timeseries_utils as tu, portfolio_utils as pu, \
    machine_learning_utils as mu

STOCK_VOL_FLOOR = 0.02


# Simulations
class MomentumSim(object):
    '''
    Stocks strategy
    '''
    def __init__(self, start_date, end_date, sample_date, universe, simulation_name, max_depth=5,
                 model_path=MODEL_PATH, load_model=False, cross_validation_buskcets=10):
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
        self.run_sim()
    
    def run_sim(self):
        logger.info('Running simulation %s' % self.simulation_name)
        self.load_universe()
        self.load_stock_data()
        #self.run_simulation()
        #self.calculate_returns()
        #self.get_analytics()
    
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
        self.r = self.stock_returns.cumsum().resample('W').last().diff()
        self.v = tu.resample(v, self.r).ffill().bfill()
        
    def create_estimation_data(self, depth):
        lookbacks = [3 ** (i+1) for i in xrange(depth)]
        return dict([('S%d' % i, self.r.ewm(span=i).mean().divide(self.v)) for i in lookbacks])

    def estimate_model(self, x, timeline, asset_returns=None, model=None):
        return mu.StockRandomBoostingComponent(x, timeline, self.asset_names, asset_returns=asset_returns,
                                               model=model, cross_validation_buckets=self.cross_validation_buckets)

    def build_model(self):
        y = self.r[self.start_date:self.end_date]
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
        self.optimal_depth = self.error_rates.index[self.error_rates == self.error_rates.min()][0]

    def calculate_signals(self):
        x = self.create_estimation_data(self.optimal_depth)
        y = self.r[self.start_date:self.end_date]
        m = self.estimate_model(x, y, model=self.model)
        self.signals = m.signals

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

    def run_simulation(self):
        self.selection = None
        self.error_rate = None
        if self.load_model:
            timeline = get_timeline(self.start_date, self.end_date).resample('W').last()
            comp = self.load_existing_model(timeline)
        elif self.cross_validation:
            self.selection, self.error_rate, comp = self.run_cross_validation(in_sample, out_of_sample)
            if self.error_rate is not None:
                logger.info('Error rate %.1f%% at %s' % (100. * self.error_rate, str(self.selection)))
        else:
            comp = self.run_without_cross_validation(in_sample, out_of_sample)
        if comp is None:
            logger.info('Failed to run model signal')
            self.model = None
        else:
            self.model = comp.model
            self.signal = comp.signal

    def calculate_returns(self):
        logger.info('Simulating strategy returns')
        rtns = []
        signals = []
        for k, v in self.stock_returns.iteritems():
            s = self.signal.get(k)
            if s is not None:
                x = v.Total / v.Vol.shift()
                x.name = k
                rtns.append(x)
                s = s.iloc[:, 0]
                s.name = k
                signals.append(s)
        rtns = pd.concat(rtns, axis=1)
        self.signals = pd.concat(signals, axis=1)
        s = tu.resample(self.signals, rtns, carry_forward=False).shift()
        self.ls_positions = get_ls_positions(s, top=30)
        self.positions = self.ls_positions.copy()
        self.positions[self.positions < 0.] = 0.
        start_date = self.start_date if self.start_date > self.positions.first_valid_index() else self.positions.first_valid_index()
        r = rtns.mul(self.positions).sum(axis=1)
        rls = rtns.mul(self.ls_positions).sum(axis=1)
        self.strategy_returns = pd.concat([r, rls], axis=1)
        self.strategy_returns.columns = ['Long Only', 'Long Short']
        self.strategy_returns = self.strategy_returns[start_date:]
        self.oos_strategy_returns = self.strategy_returns[self.sample_date:]

    def get_analytics(self):
        logger.info('Calculating analytics')
        self.analytics = pu.get_returns_analytics(self.strategy_returns)
        self.oos_analytics = pu.get_returns_analytics(self.oos_strategy_returns)
        