'''
Created on 8 Oct 2017

@author: wayne
'''
import numpy as np
import pandas as pd
from scipy import optimize as sop
from quant.lib.main_utils import logger


def get_weight_matrix(w):
    ans = np.array(w)
    if ans.ndim == 1:
        ans = np.array([ans])
    if np.size(ans, 0) == 1 and np.size(ans, 1) > 1:
        ans = ans.T
    ans[np.isinf(ans)] = 0.
    ans[np.isnan(ans)] = 0.
    return ans


def get_weight_dataframe(w):
    if isinstance(w, pd.DataFrame):
        ans = w
    else:
        ans = w.to_frame()
    if len(ans) == 1 and len(ans.columns) > 1:
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
    

class MeanVarianceOptimizer(object):
    
    def __init__(self, instrument, factor_loadings=None, factor_covariance=None, specific_covariance=None,
                 stock_alpha=None, start_weight=None, existing_stocks=None, factor_lambda=1,
                 specific_lambda=1., instrument_bounds=None, *args, **kwargs):
        self.instrument = instrument
        self.factor_loadings = factor_loadings
        self.factor_covariance = factor_covariance
        self.specific_covariance = specific_covariance
        self.stock_alpha = stock_alpha
        self.start_weight = start_weight
        self.existing_stocks = existing_stocks
        self.factor_lambda = factor_lambda
        self.specific_lambda = specific_lambda
        self.instrument_bounds = instrument_bounds
        self.run()

    def run(self):
        self.format_input()
        self.run_optimization()

    def format_input(self):
        stocks = []
        factors = []
        ins = get_weight_dataframe(self.instrument)
        stocks += list(ins.columns)
        instruments = sorted(list(ins.index))
        if self.existing_stocks is None:
            stk = pd.DataFrame(np.zeros((len(ins.columns), 1)), index=ins.columns)
        else:
            stk = get_weight_dataframe(self.existing_stocks)
        stocks += list(stk.index)
        if self.factor_loadings is not None and self.factor_covariance is not None:
            stocks += list(self.factor_loadings.columns)
            factors += list(self.factor_loadings.index)
            factors += list(self.factor_covariance.columns)
            factors = sorted(list(set(factors)))
            f = self.factor_loadings.loc[factors].fillna(0.)
            c = self.factor_covariance.loc[factors, factors].fillna(0.)
            sf = pd.DataFrame(np.dot(f.T, np.dot(c, f)), index=f.columns, columns=f.columns)
        else:
            sf = None
        if self.specific_covariance is not None:
            stocks += list(self.specific_covariance.columns)
            s = self.specific_covariance
        else:
            s = pd.DataFrame([[0.]])
        if self.stock_alpha is not None:
            alpha = get_weight_dataframe(self.stock_alpha)
            stocks += list(alpha.index)
        else:
            alpha = None
        if self.instrument_bounds is not None:
            b = self.instrument_bounds
        else:
            b = None
        self.instrument_names = instruments
        self.stock_names = sorted(list(set(stocks)))
        self.factors = factors
        if self.start_weight is None:
            self._w = pd.DataFrame(np.zeros((len(self.instrument_names), 1)), index=self.instrument_names)
        else:
            self._w = get_weight_dataframe(self.start_weight).loc[self.instrument_names].fillna(0.)
        self._ins = ins.loc[self.instrument_names, self.stock_names].fillna(0.)
        self._stocks = stk.loc[self.stock_names].fillna(0.)
        if alpha is None:
            self._m = pd.DataFrame(np.zeros((len(self.stock_names), 1)), index=self.stock_names)
        else:
            self._m = alpha.loc[self.stock_names].fillna(0.)
        if sf is None:
            self._sf = pd.DataFrame(np.zeros((len(self.stock_names), len(self.stock_names))),
                                    index=self.stock_names, columns=self.stock_names)
        else:
            self._sf = sf.loc[self.stock_names, self.stock_names].fillna(0.)
        if s is None:
            self._s = pd.DataFrame(np.zeros((len(self.stock_names), len(self.stock_names))),
                                   index=self.stock_names, columns=self.stock_names)
        else:
            self._s = s.loc[self.stock_names, self.stock_names].fillna(0.)
        if b is None:
            self._ib = []
        else:
            b.loc[self.instrument_names, ['Lower', 'Upper']].fillna(pd.Series([-1e8, 1e8], index=['Lower', 'Upper']))
            self._ib = [tuple(b.loc[idx]) for idx in self.instrument_names]

    def run_optimization(self):
        
        def obj(w, m, ins, stocks, sf, lambda_f, s, lambda_s, *args, **kwargs):
            ww = get_weight_matrix(w)
            portfolio = Portfolio(ww, ins, stocks)
            x = portfolio.result()
            variance_f = Variance(x, sf)
            variance_s = Variance(x, s)
            return - np.dot(x.T, m)[0][0] + lambda_f * variance_f.result() + lambda_s * variance_s.result()
        
        def obj_prime(w, m, ins, stocks, sf, lambda_f, s, lambda_s, *args, **kwargs):
            ww = get_weight_matrix(w)
            portfolio = Portfolio(ww, ins, stocks)
            x = portfolio.result()
            variance_f = Variance(x, sf)
            variance_s = Variance(x, s)
            xp = portfolio.prime()
            ans = -np.dot(xp, m) + lambda_f * np.dot(xp, variance_f.prime()) + lambda_s * np.dot(xp, variance_s.prime())
            return ans.T[0]
        
        args = (self._m, self._ins, self._stocks, self._sf, self.factor_lambda, self._s, self.specific_lambda)
        out, fx, its, imode, smode = sop.fmin_slsqp(func=obj, x0=self._w.values.flatten(), bounds=self._ib,
                                                    fprime=obj_prime, args=args, disp=0, full_output=True)
        self.solution = pd.DataFrame(np.reshape(out, (len(self.instrument_names), 1)), index=self.instrument_names)
        self._final_obj = fx
        self._iterations = its
        self._imode = imode
        self._smode = smode

        
        
            