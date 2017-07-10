'''
Created on 23 Jun 2017

@author: wayne
'''
import numpy as np
import pandas as pd
from quant.lib import timeseries_utils as tu
from quant.lib.main_utils import logger

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


def pandas_ewma(data, span=None):
    return data.ewm(span=span, ignore_na=True).mean() if span > 1. else data


def pandas_weeks_ewma(data, span=None):
    df = data.index.to_series().diff().dropna().mean()
    es = 7. * span / df.days if span is not None else None
    return data.ewm(span=es, ignore_na=True).mean() if es > 1. else data 


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
    '''
    Stump estimation of a single iteration. 
    '''
    data = pd.concat([x, y], axis=1).copy().values
    data = data[np.argsort(x.values)]
    tot = np.sum(np.abs(y))
    if tot == 0.:
        tot = 1.
    pos = data[:, 1].copy() / tot
    pos[pos < 0] = 0.
    pos = np.cumsum(pos)
    neg = data[:, 1].copy() / tot
    neg[neg > 0] = 0.
    neg = np.cumsum(neg[::-1])
    neg = np.array(list(neg[:-1])[::-1] + [0.])
    loc = np.argsort(np.abs(pos - neg - 0.5))[-1]
    return np.mean(data[loc:loc+2, 0]) if loc < len(data)-1 else 1. + data[-1, 0]


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
        alpha[y < 0] *= np.exp(w)
        alpha[y > 0] *= np.exp(-w)
        alpha /= np.sum(alpha)
        ans.append([BOOSTING_INTERCEPT, y.name, np.nan, e, w])
    for i in np.arange(np.size(x, 1)):
        pred = x.iloc[:, i]
        u = DummyStump(pred, y * alpha)
        e = StumpError(pred, y * alpha, u)
        w = get_weight_from_error(e)
        alpha[(pred < u) & (y > 0)] *= np.exp(w)
        alpha[(pred > u) & (y < 0)] *= np.exp(w)
        alpha[(pred < u) & (y < 0)] *= np.exp(-w)
        alpha[(pred > u) & (y > 0)] *= np.exp(-w)
        alpha /= np.sum(alpha)
        ans.append([pred.name, y.name, u, e, w])
    return pd.DataFrame(ans, columns=['predicative', 'target', 'estimation', 'Error', 'weight'])


def BoostingStump(x, y, estimate_intercept=True, no_of_variables=None, *args, **kwargs):
    '''
    Boosting stump with Pandas variables
    
    Input
    --------
    x                     Pandas DataFrame or Series of predictive variables
    y                     Pandas DataFrame or Series of target variables
    estimate_intercept    Boolean
    no_of_variables       integer, the number of variables to use

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


def RandomBoosting(x, y, forest_size=100, estimate_intercept=True, no_of_variables=None, *args, **kwargs):
    '''
    Random forest on top of boosting
    Returns a list of tuples of (randomization, boosting model)
    '''
    myx, _ = give_me_pandas_variables(x, y)
    sequence = get_random_sequence(np.size(myx, 1), forest_size)
    return [(seq, BoostingStump(x.iloc[:, seq], y, estimate_intercept, no_of_variables)) for seq in sequence]
    
    
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
    return 2. * (x >= c) - 1.


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
            else:
                tmp = StumpPrediction(x[predicative], estimation)
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
                 prediction_function, params=None):
        self.asset_returns = asset_returns
        self.in_sample_data = in_sample_data
        self.out_of_sample_data = out_of_sample_data
        self.model_function = model_function
        self.prediction_function = prediction_function
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
        self._x = in_sample_data.fillna(in_sample_data.median(axis=0))
        self._y = self.asset_returns.fillna(0.)
        self._z = out_of_sample_data.fillna(out_of_sample_data.median(axis=0))

    def estimate_model(self):
        self.model = self.model_function(x=self._x, y=self._y, **self.params)
    
    def calculate_signals(self):
        in_sample_signal = self.prediction_function(x=self._x, ans=self.model)
        signal_distribution = tu.get_distribution_parameters(in_sample_signal)
        self.signal = self.prediction_function(x=self._z, ans=self.model)
        self.normalized_signal = tu.get_distribution_scores(self.signal, signal_distribution)


class BoostingStumpComponent(object):
    def __init__(self, asset_returns, in_sample_data, out_of_sample_data, params=None):
        self.core = Component(asset_returns, in_sample_data, out_of_sample_data,
                              model_function=BoostingStump, prediction_function=BoostingPrediction,
                              params=params)
        self.model = self.core.model
        self.signal = self.core.signal
        self.normalized_signal = self.core.normalized_signal


class RandomBoostingComponent(object):
    def __init__(self, asset_returns, in_sample_data, out_of_sample_data, params=None):
        self.core = Component(asset_returns, in_sample_data, out_of_sample_data,
                              model_function=RandomBoosting, prediction_function=RandomBoostingPrediction,
                              params=params)
        self.model = self.core.model
        self.signal = self.core.signal
        self.normalized_signal = self.core.normalized_signal
