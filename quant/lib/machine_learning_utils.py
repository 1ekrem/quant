'''
Created on 23 Jun 2017

@author: wayne
'''
import numpy as np
import pandas as pd
from scipy import stats as ss
from quant.lib import timeseries_utils as tu
from quant.lib.main_utils import logger
from audioop import cross

MINIMUM_ERROR = 1e-4
BOOSTING_INTERCEPT = 'Intercept'


# utility functions
def give_me_pandas_variables(x, y):
    '''
    Handle all inputs of x and y variables for machine learning.
    Reformat and fill nan with 0.

    Input
    --------
    x    Input variable
    y    Target variable

    '''
    assert isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)
    assert isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)
    
    myx = x.to_frame() if isinstance(x, pd.Series) else x
    myy = y.to_frame() if isinstance(y, pd.Series) else y
    
    assert len(x) == len(y)
    return myx.fillna(0.), myy.fillna(0.)


def data_to_score(data, mean=None, sd=None):
    ans = data.copy()
    d = data[-np.isnan(data)]
    if mean is None:
        mean = np.mean(d)
    if sd is None:
        sd = np.std(d)
    ans[-np.isnan(ans)] = ss.norm.cdf(ans.values, loc=mean, scale=sd)
    return 2. * (ans - .5)


# Boosting
def positive_sum(x):
    '''
    Sum of positive values
    '''
    return np.sum(x[x>0])


def negative_sum(x):
    '''
    Sum of negative values
    '''
    return np.sum(x[x<0])


def StumpError(x, y, c):
    '''
    Error rate of stump prediction
    Error rate = Total value of wrong predictions / Total possible predictions
    '''
    error = (positive_sum(y[x<c]) - negative_sum(y[x>=c]))
    total = (positive_sum(y) - negative_sum(y))
    return error / total if total > 0 and not np.isnan(c) else np.nan


def DummyStump(x, y):
    data_x = np.array(x).flatten()
    data_y = np.array(y).flatten()
    tot = np.sum(np.abs(data_y))
    if tot > 0:
        data_y /= tot 
    data = np.array(zip(data_x, data_y), dtype=[('x', np.float64), ('y', np.float64)])
    data = np.sort(data, order='x')
    pos = data['y'].copy()
    pos[pos < 0.] = 0.
    pos = np.cumsum(pos)
    neg = data['y'].copy()
    neg[neg > 0.] = 0.
    neg = np.append(np.cumsum(neg[::-1])[:-1][::-1], 0.)
    score = np.abs(pos - neg - 0.5)
    loc = np.arange(len(data_x))
    loc = loc[score == np.max(score)][0]
    cutoff = data['x'][loc] if loc > 0 else np.nan
    ploc = 1. * loc / len(data_x)
    if ploc > .5:
        ploc = 1. - ploc
    score = 2. * score[loc]
    return cutoff, ploc, score


def get_weight_from_error(e):
    '''
    Calculate weight given error rate in boosting stump.
    '''
    return .5 * np.log((1. - e + MINIMUM_ERROR) / (e + MINIMUM_ERROR))

    
def estimate_boosting_stump(x, y, estimate_intercept=True):
    '''
    Estimating boosting stump for one target variable
    '''
    ans = []
    alpha = np.repeat(1. / len(y), len(y))
    if estimate_intercept:
        pred = x.iloc[:, 0]
        e = StumpError(pred, y * alpha, np.min(pred))
        w = get_weight_from_error(e)
        alpha *= 1. * (y < 0) * np.exp(w) + 1. * (y >= 0) * np.exp(-w)
        alpha /= np.sum(alpha)
        ans.append([BOOSTING_INTERCEPT, y.name, np.nan, e, w, np.nan, np.nan])
    for i in np.arange(np.size(x, 1)):
        pred = x.iloc[:, i]
        u, ploc, score = DummyStump(pred, y * alpha)
        uu = np.min(pred) - 1. if np.isnan(u) else u
        e = StumpError(pred, y * alpha, uu)
        w = get_weight_from_error(e)
        alpha_delta = 1. * (pred < uu) * (y >= 0) * np.exp(w)
        alpha_delta += 1. * (pred >= uu) * (y < 0) * np.exp(w)
        alpha_delta += 1. * (pred < uu) * (y < 0) * np.exp(-w)
        alpha_delta += 1. * (pred >= uu) * (y >= 0) * np.exp(-w)
        alpha *= alpha_delta
        alpha /= np.sum(alpha)
        ans.append([pred.name, y.name, u, e, w, ploc, score])
    return pd.DataFrame(ans, columns=['predicative', 'target', 'estimation', 'Error', 'weight', 'pLoc', 'score'])


def BoostingStump(x, y, estimate_intercept=True, no_of_variables=None, *args, **kwargs):
    '''
    Boosting stump with Pandas variables
    
    Input
    --------
    x                           Pandas DataFrame or Series of predictive variables
    y                           Pandas DataFrame or Series of target variables
    estimate_intercept          Boolean
    no_of_variables             integer, the number of variables to use

    Output
    --------
    ans    Pandas DataFrame with predictive variable name, target variable name,
            stump decision point of each x variable and weight corresponding to each x variable
    
    Notes
    --------
    The target is to maximize the success rate of predicting the sign of y in-sample, weighted by the magnitude of y.
    Use caution to normalize y before estimation.
    x variables are used in the order that they are given.
    If there are multiple columns in y, each column will be predicted independently.

    '''
    myx, myy = give_me_pandas_variables(x, y)
    if no_of_variables is not None:
        if np.size(myx, 1) > no_of_variables:
            myx = myx.iloc[:, :no_of_variables]
    ans = []
    for i in xrange(np.size(myy, 1)):
        try:
            ans.append(estimate_boosting_stump(myx, myy.iloc[:, i], estimate_intercept))
        except Exception as e:
            logger.warning('Boosting stump failed at variable %s: %s' % (str(myy.columns[i]), str(e)))
    return pd.concat(ans, axis=0)


def get_random_sequence(n_variables, forest_size, seed=0):
    '''
    Returns a set of random sequences of variable orders
    '''
    np.random.seed(seed)
    ans = []
    order = np.arange(n_variables)
    for i in xrange(forest_size):
        np.random.shuffle(order)
        ans.append(order.copy())
    return ans


def get_cross_validation_buckets(data_size, buckets):
    seq = get_random_sequence(data_size, 1)[0]
    step_size = 1. * data_size / buckets
    ans = []
    for i in xrange(buckets):
        lower = np.int(np.round(i * step_size))
        upper = np.int(np.round((i+1) * step_size)) if i < buckets - 1 else data_size
        ans.append(seq[lower:upper])
    return ans


def RandomBoosting(x, y, forest_size=100, estimate_intercept=True, no_of_variables=None, *args, **kwargs):
    '''
    Random forest on top of boosting
    Returns a list of tuples of (randomization, boosting model)
    '''
    myx, _ = give_me_pandas_variables(x, y)
    sequence = get_random_sequence(np.size(myx, 1), forest_size)
    return [(seq, BoostingStump(myx.iloc[:, seq], y, estimate_intercept, no_of_variables)) for seq in sequence]
    

# Prediction
def StumpPrediction(x, c):
    '''
    Prediction with a single stump

    Input
    --------
    x    Input variable
    c    Cut-off point

    Notes
    --------
    Predict 1 if x >=c else -1

    '''
    cc = np.min(x) - 1. if np.isnan(c) else c
    return 2. * (x >= cc) - 1.


def BoostingPrediction(x, ans):
    '''
    Boosting prediction
    
    Input
    --------
    x        Input variable
    ans      Boosting estimation results

    Output
    --------
    DataFrame of predictions for the target variables

    '''
    if ans is None:
        return None
    else:
        targets = list(set(ans['target'].values))
        predictions = dict(zip(targets, [[]] * len(targets)))
        for config in ans.T.to_dict().values():
            target = config['target']
            predicative = config['predicative']
            weight = config['weight']
            estimation = config['estimation']
            if predicative == BOOSTING_INTERCEPT:
                tmp = x.iloc[:, 0].copy() * np.nan
                tmp = tmp.fillna(1.)
            elif predicative in x.columns:
                tmp = StumpPrediction(x[predicative], estimation)
            else:
                tmp = x.iloc[:, 0].copy() * np.nan
                tmp = tmp.fillna(0.)
            predictions[target].append(tmp * weight)
        results = []
        for k, v in predictions.iteritems():
            tmp = pd.concat(v, axis=1).sum(axis=1)
            tmp.name = k
            results.append(tmp)
        return pd.concat(results, axis=1) 


def RandomBoostingPrediction(x, ans):
    '''
    Prediction with random boosting model
    '''
    df_concat = pd.concat([BoostingPrediction(x.iloc[:, seq], model) for seq, model in ans])
    return df_concat.groupby(df_concat.index).mean()


# Strategy Components
class Component(object):
    '''
    Strategy component
    
    Input
    --------
    asset_returns
    in_sample_data
    out_of_sample_data
    model_function
    prediction_function
    use_score

    '''    
    def __init__(self, asset_returns, in_sample_data, out_of_sample_data, model_function, 
                 prediction_function, params=None, model=None, *args, **kwargs):
        self.asset_returns = asset_returns
        self.in_sample_data = in_sample_data
        self.out_of_sample_data = out_of_sample_data
        self.model_function = model_function
        self.prediction_function = prediction_function
        self.model = model
        self.params = {} if params is None else params
        self.run_model()

    def run_model(self):
        self.prepare_data()
        self.estimate_model()
        self.calculate_signals()
    
    def prepare_data(self):
        self.distribution_params = tu.get_distribution_parameters(self.in_sample_data)
        in_sample_data = self.in_sample_data
        out_of_sample_data = self.out_of_sample_data
        medians = in_sample_data.median(axis=0)
        self._x = in_sample_data.fillna(medians)
        self._y = self.asset_returns.fillna(0.)
        self._z = out_of_sample_data.fillna(medians)

    def estimate_model(self):
        if self.model is None:
            self.model = self.model_function(x=self._x, y=self._y, **self.params)
    
    def calculate_signals(self):
        in_sample_signal = self.prediction_function(x=self._x, ans=self.model)
        signal_distribution = tu.get_distribution_parameters(in_sample_signal)
        self.signal = self.prediction_function(x=self._z, ans=self.model)
        self.normalized_signal = tu.get_distribution_scores(self.signal, signal_distribution)


class BoostingStumpComponent(object):
    def __init__(self, asset_returns, in_sample_data, out_of_sample_data, params=None, model=None):
        self.core = Component(asset_returns, in_sample_data, out_of_sample_data,
                              model_function=BoostingStump, prediction_function=BoostingPrediction,
                              params=params, model=model)
        self.model = self.core.model
        self.signal = self.core.signal
        self.normalized_signal = self.core.normalized_signal


class RandomBoostingComponent(object):
    def __init__(self, asset_returns, in_sample_data, out_of_sample_data, params=None, model=None):
        self.core = Component(asset_returns, in_sample_data, out_of_sample_data,
                              model_function=RandomBoosting, prediction_function=RandomBoostingPrediction,
                              params=params, model=model)
        self.model = self.core.model
        self.signal = self.core.signal
        self.normalized_signal = self.core.normalized_signal


class CrossSectionComponent(object):    
    def __init__(self, predictors, timeline, asset_names, model_function, prediction_function,
                 asset_returns=None, model=None, cross_validation_buckets=10, params=None, *args, **kwargs):
        self.asset_returns = asset_returns
        self.predictors = predictors
        self.timeline = timeline
        self.asset_names = asset_names
        self.model_function = model_function
        self.prediction_function = prediction_function
        self.model = model
        self.cross_validation_buckets = cross_validation_buckets
        self.params = {} if params is None else params
        self.run_model()

    def _shape(self, data):
        return tu.resample(data, self.timeline).loc[:, self.asset_names]

    def run_model(self):
        self.prepare_data()
        self.estimate_model()
        self.calculate_signals()
    
    def prepare_data(self):
        if self.asset_returns is not None:
            self._y = tu.dataframe_to_series(self._shape(self.asset_returns))
        self._x = pd.concat([tu.dataframe_to_series(self._shape(v)) for v in self.predictors.values()], axis=1).fillna(0.)
        self._x.columns = self.predictors.keys()

    def estimate_model(self):
        if self.model is None and self._y is not None:
            logger.info('Estimating model with %d observations' % len(self._x))
            self.model = self.model_function(x=self._x.loc[~self._y.isnull(), :], y=self._y[~self._y.isnull()], **self.params)
    
    def calculate_signals(self):
        self.signals = tu.series_to_dataframe(self.prediction_function(x=self._x, ans=self.model), self.asset_names, self.timeline.index)

    def run_cross_validation(self):
        if self._y is not None:
            y = self._y[~self._y.isnull()]
            x = self._x[~self._y.isnull()]
            self._seq = get_cross_validation_buckets(len(y), self.cross_validation_buckets)
            self.errors = []
            for i, s in enumerate(self._seq):
                logger.info('Cross validation bucket %s' % (i + 1))
                x_out = x.iloc[s]
                y_out = y.iloc[s]
                x_in = x.loc[~x.index.isin(x_out.index)]
                y_in = y.loc[~y.index.isin(y_out.index)]
                m = self.model_function(x=x_in, y=y_in, **self.params)
                pred = self.prediction_function(x=x_out, ans=m)
                self.errors.append(StumpError(pred.values.flatten(), y_out.values.flatten(), 0.))
            self.error_rate = np.mean(self.errors)
        else:
            self.errors = None
            self.error_rate = None

class StockRandomBoostingComponent(object):
    def __init__(self, predictors, timeline, asset_names, asset_returns=None, model=None, cross_validation_buckets=10):
        self.core = CrossSectionComponent(predictors, timeline, asset_names, asset_returns=asset_returns, model=model,
                                          model_function=RandomBoosting, prediction_function=RandomBoostingPrediction,
                                          cross_validation_buckets=cross_validation_buckets)
        self.model = model
        self.run()

    def run(self):
        if self.model is None:
            self.core.estimate_model()
            self.model = self.core.model
        if self.model is not None:
            self.core.calculate_signals()
            self.signals = self.core.signals

    def run_cross_validation(self):
        self.core.run_cross_validation()
        self.errors = self.core.errors
        self.error_rate = self.core.error_rate
