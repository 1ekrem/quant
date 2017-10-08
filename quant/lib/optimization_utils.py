'''
Created on 8 Oct 2017

@author: wayne
'''
import numpy as np
import pandas as pd
from quant.lib.main_utils import logger


def get_weight_matrix(w):
    ans = np.array(w)
    if ans.ndim == 1:
        ans = np.array([ans])
    if np.size(ans, 0) == 1:
        ans = ans.T
    ans[np.isinf(ans)] = 0.
    ans[np.isnan(ans)] = 0.
    return ans


def get_weight_dataframe(w):
    if isinstance(w, pd.DataFrame):
        ans = w
    else:
        ans = w.to_frame()
    if len(ans) == 1:
        ans = ans.T
    ans[np.isinf(ans)] = 0.
    ans[np.isnan(ans)] = 0.
    return ans


class Portfolio(object):
    
    def __init__(self, w, instrument, stocks, *args, **kwargs):
        self.w = get_weight_matrix(w)
        self.instrument = get_weight_matrix(instrument)
        self.stocks = get_weight_matrix(stocks)

    def result(self):
        return self.stocks + np.dot(self.instrument.T, self.w)

    def prime(self):
        return self.instrument


class Variance(object):

    def __init__(self, w, v, *args, **kwargs):
        self.w = get_weight_matrix(w)
        self.v = get_weight_matrix(v)
    
    def result(self):
        return np.dot(self.w.T, np.dot(self.v, self.w))[0][0]

    def prime(self):
        return 2. * np.dot(self.v, self.w)


class GradientDescentOptimizer(object):
    
    def __init__(self, instrument, stocks, stock_covariance, start_weight=None,
                 epsilon=1e-2, max_iteration=5e3, speed=0.1, logging=True, *args, **kwargs):
        self.start_weight = start_weight
        self.instrument = instrument
        self.stocks = stocks
        self.stock_covariance = stock_covariance
        self.epsilon = epsilon
        self.max_iteration = max_iteration
        self.speed = speed
        self.logging = logging
        self.run()

    def run(self):
        self.format_input()
        self.get_covariance()
        self.run_gradient_descent()

    def format_input(self):
        ins = get_weight_dataframe(self.instrument)
        stk = get_weight_dataframe(self.stocks)
        cov = get_weight_dataframe(self.stock_covariance)
        w = get_weight_dataframe(pd.Series([0.] * len(ins), index=ins.index) if self.start_weight is None else self.start_weight)
        self.stock_names = sorted(list(set(list(ins.columns) + list(stk.index) + list(cov.columns))))
        self.instrument_names = sorted(list(set(list(ins.index) + list(w.index))))
        self._w = w.loc[self.instrument_names].fillna(0.)
        self._ins = ins.loc[self.instrument_names, self.stock_names].fillna(0.)
        self._stocks = stk.loc[self.stock_names].fillna(0.)
    
    def get_covariance(self):
        self._cov = self.stock_covariance.loc[self.stock_names, self.stock_names].fillna(0.)
    
    def objective(self, variance):
        ans = variance.result()
        return ans
    
    def objective_prime(self, variance, portfolio):
        ans = np.dot(portfolio.prime(), variance.prime())
        return ans
    
    def calc_iteration(self, w):
        portfolio = Portfolio(w, self._ins, self._stocks)
        variance = Variance(portfolio.result(), self._cov)
        obj = self.objective(variance)
        obj_prime = self.objective_prime(variance, portfolio)
        step = np.sqrt(np.sum(obj_prime ** 2))
        return obj, pd.DataFrame(obj_prime, index=w.index), \
            pd.DataFrame(portfolio.result(), index=self._stocks.index), step

    def run_gradient_descent(self):
        w = self._w
        obj, obj_prime, stock_weights, step = self.calc_iteration(w)
        count = 0
        ws = [w.copy()]
        ss = [stock_weights.copy()]
        objs = [obj]
        obj_primes = [obj_prime]
        steps = [step]
        while count < self.max_iteration and step > self.epsilon:
            w -= self.speed * obj_prime
            obj, obj_prime, stock_weights, step = self.calc_iteration(w)
            count += 1
            ws.append(w.copy())
            ss.append(stock_weights.copy())
            objs.append(obj)
            obj_primes.append(obj_prime)
            steps.append(step)
            if self.logging:
                logger.info('Count: %d Step: %.2f' % (count, step))
        self.ws = pd.concat(ws, axis=1).T
        self.ss = pd.concat(ss, axis=1).T
        self.objs = pd.Series(objs, name='Objective')
        self.obj_primes = pd.concat(obj_primes, axis=1).T
        self.steps = pd.Series(steps, name='Step')
        self.ws.index = self.objs.index
        self.ss.index = self.objs.index
        self.obj_primes.index = self.objs.index
    
        
        
        
