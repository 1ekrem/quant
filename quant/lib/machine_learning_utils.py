'''
Created on 23 Jun 2017

@author: wayne
'''
import numpy as np
import pandas as pd
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

    
def estimate_boosting_stump(x, y):
    '''
    Estimating boosting stump for one target variable
    '''
    ans = []
    alpha = np.repeat(1. / len(y), len(y))
    pred = x.iloc[:, 0]
    e = StumpError(pred, y * alpha, np.min(pred))
    w = get_weight_from_error(e)
    alpha[y < 0] *= np.exp(w)
    alpha[y > 0] *= np.exp(-w)
    alpha /= np.sum(alpha)
    ans.append([BOOSTING_INTERCEPT, y.name, np.nan, w])
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
        ans.append([pred.name, y.name, u, w])
    return pd.DataFrame(ans, columns=['predicative', 'target', 'estimation', 'weight'])


def BoostingStump(x, y):
    '''
    Boosting stump with Pandas variables
    
    Input
    --------
    x    Pandas DataFrame or Series of predictive variables
    y    Pandas DataFrame or Series of target variables

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
    ans = []
    for i in xrange(np.size(myy, 1)):
        try:
            ans.append(estimate_boosting_stump(myx, myy.iloc[:, i]))
        except Exception as e:
            logger.warning('Boosting stump failed at variable %s: %s' % (str(myy.columns[i]), str(e)))
    return pd.concat(ans, axis=0)


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

