'''
Created on Aug 23, 2016

@author: Wayne
'''
import numpy as np
import pandas as pd
import cPickle as pickle
from datetime import datetime as dt
from matplotlib import pyplot as plt
from lib import timeseries_utils as tu
from research import econ

np.random.seed(0)
PATH = 'C:/Users/Wayne/Documents/'


def resample(ts, timeline, carry_forward=True):
    index = set(ts.index)
    idx = set([x for x in timeline.index if x not in index])
    if len(idx)>0:
        if isinstance(ts, pd.Series):
            tmp = pd.Series([np.nan] * len(idx), index=idx, name=ts.name)
        else:
            tmp = pd.DataFrame(np.ones((len(idx), np.size(ts,1))) * np.nan, index=idx, columns=ts.columns)
        ts = pd.concat([ts, tmp], axis=0).sort_index()
        if carry_forward:
            for t in range(len(ts)-1):
                if ts.index[t+1] in idx:
                    ts.iloc[t+1] = ts.iloc[t]
    return ts.loc[timeline.index]


def read_price_data(tab='EQ'):
    data = pd.read_excel('C:\Users\Wayne\Documents\data\data.xlsx', tab)
    ans = []
    i = 1
    while i<len(data.columns):
        ans.append(pd.Series(data.values[:, i+1], index=data.values[:, i], name=data.columns[i+1]).dropna())
        i +=2
    return pd.concat(ans, axis=1)


def get_input_data():
    p = read_price_data()
    p = p.resample('w', 'last')
    return (p.diff() / p.shift()).dropna()


def gradientDescent(x, y, theta, alpha, m, numIterations, epsilon, penalty):
    xTrans = x.transpose()
    i = 0
    step = 1.
    while step > epsilon and i < numIterations:
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = (np.sum(loss ** 2) + penalty * np.sum(theta ** 2)) / (2 * m) 
        gradient = (np.dot(xTrans, loss) + penalty * theta) / m
        step = np.sqrt(np.sum(gradient ** 2))
        theta = theta - alpha * gradient
        i += 1
    return theta


def estimate_gradient_descent(numIterations=1e5, alpha=1e-2, epsilon=1e-4, penalty=0.):
    data = get_input_data()
    y = data.values[:, 0]
    x = np.hstack([np.ones((len(data), 1)), data.values[:, 1:]])
    m, n = np.shape(x)
    theta = np.zeros(n)
    ans = gradientDescent(x, y, theta, alpha, m, numIterations, epsilon, penalty)
    return pd.Series(ans, index=['Const'] + list(data.columns[1:]))


def run_cross_validation(y, x, n, penalties = np.arange(5e-2, .5, 5e-2)):
    assert len(y)>n and len(x)>n and len(y) == len(x)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    space = 1. * len(y) / n
    errors = []
    for i in xrange(n):
        low = np.int(space * i)
        high = np.int(space * (i+1))
        if i == n-1:
            high = len(y)
        idxt = idx[low:high]
        idxs = list(idx[:low]) + list(idx[high:])
        xs = x[idxs]
        ys = y[idxs]
        xt = x[idxt]
        yt = y[idxt]
        m, v = np.shape(xs)
        mt, _ = np.shape(xt)
        pp = []
        for p in penalties:
            theta = np.zeros(v)
            theta = gradientDescent(xs, ys, theta, 1e-2, m, 1e5, 1e-4, p)
            hypothesis = np.dot(xt, theta)
            loss = hypothesis - yt
            cost = np.sum(loss ** 2) / (2 * mt)
            pp.append(cost)
        errors.append(pp)
    errors = np.array(errors)
    me = np.mean(errors, axis=0)
    tp = penalties[np.argsort(me)[0]]
    theta = np.zeros(v)
    theta = gradientDescent(x, y, theta, 1e-2, len(x), 1e5, 1e-4, tp)
    return theta, tp, me


def estimate_gradient_decent_with_cross_validation(n=10, penalties = np.arange(5e-3, 5e-2, 5e-3)):
    data = get_input_data()
    y = data.values[:, 0]
    x = np.hstack([np.ones((len(data), 1)), data.values[:, 1:]])
    theta, tp, me = run_cross_validation(y, x, n, penalties)
    ans = pd.Series(theta, index=['Const'] + list(data.columns[1:]))
    me = pd.Series(me, index = penalties)
    return ans, tp, me


def get_probability(w, x):
    return 1. / (1. + np.exp(-np.sum(w * x, axis=1)))


def gradientAscent(x, y, theta, alpha, numIterations, epsilon, penalty):
    i = 0
    step = 1.
    prob = []
    while step > epsilon and i < numIterations:
        gradient = np.sum(np.reshape(.5 * (np.sign(y) + 1) - get_probability(theta, x), (len(y), 1)) * x, axis=0) - 2. * penalty * theta
        step = np.sqrt(np.sum(gradient ** 2))
        theta = theta + alpha * gradient
        i += 1
        prob.append(np.sum(y * (get_probability(theta, x) - .5)))
    return theta, prob


def estimate_gradient_ascent_with_cross_validation(x, y, n, alpha, numIterations, epsilon):
    np.random.seed(0)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    step = 1. * len(y) / n
    e = []
    ls = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.]
    for l in ls:
        tmp = []
        for i in np.arange(n):
            x0 = np.int(step * i)
            x1 = len(y) if i == n-1 else np.int(step * (i+1))
            xs = np.vstack([x[:x0], x[x1:]])
            ys = np.array(list(y[:x0]) + list(y[x1:]))
            xv = x[x0:x1]
            yv = y[x0:x1]
            theta, _ = gradientAscent(xs, ys, np.zeros(np.size(x, 1)), alpha, numIterations, epsilon, l)
            ans = np.sign(get_probability(theta, xv) - .5) * yv
            tmp.append(1. * np.sum(ans < 0) / len(yv))
        e.append(np.mean(tmp))
    lx = ls[np.argsort(e)[0]]
    theta, prob = gradientAscent(x, y, np.zeros(np.size(x, 1)), alpha, numIterations, epsilon, lx)
    return theta, prob, lx, np.array(e)


def positive_sum(x):
    return np.sum(x[x>0])


def negative_sum(x):
    return np.sum(x[x<0])


def StumpPrediction(x, c):
    return np.sign(x - c)


def LogisticPrediction(x, theta):
    return np.sign(get_probability(theta, x) - .5)


def StumpError(x, y, c):
    return (positive_sum(y[x<c]) - negative_sum(y[x>c])) / (positive_sum(y) - negative_sum(y))


def ValidationPrediction(x, cs):
    return np.sum(np.vstack([(StumpPrediction(x[:, i - 1], c[0]) if i>0 else np.ones(len(x))) * c[1] for i, c in enumerate(cs)]), axis=0)


def StumpValidationError(x, y, cs):
    p = ValidationPrediction(x, cs)
    return StumpError(p, y, 0)


def LogisticValidationPrediction(x, cs):
    return np.sum(np.vstack([LogisticPrediction(x[i], c[0]) * c[1] for i, c in enumerate(cs)]), axis=0)


def LogisticValidationError(x, y, cs):
    p = LogisticValidationPrediction(x, cs)
    return StumpError(p, y, 0)


def UnitStump(x, y):
    return np.min(x) -1.


def backupDummyStump(x, y):
    l = np.argsort(x)
    tot = positive_sum(y) - negative_sum(y)
    c = np.abs(np.array([positive_sum(y[l[:i]])-negative_sum(y[l[i:]]) for i in np.arange(len(y))]) / tot - .5)
    loc = np.argsort(c)[-1]
    return np.mean(x[[l[loc-1], l[loc]]]) if loc >0 else x[l[loc]] - 1.


def DummyStump(x, y):
    data = np.vstack([x, y]).T[np.argsort(x)]
    tot = np.sum(np.abs(y))
    pos = 1. * data[:, 1] / tot
    pos[pos < 0] = 0.
    pos = np.cumsum(pos)
    neg = 1. * data[:, 1] / tot
    neg[neg > 0] = 0.
    neg = np.cumsum(neg[::-1])
    neg = np.array(list(neg[:-1])[::-1] + [0.])
    loc = np.argsort(np.abs(pos - neg - 0.5))[-1]
    return np.mean(data[loc:loc+2, 0]) if loc < len(data)-1 else 1. + data[-1, 0]


def BoostingStump(x, y):
    ans = []
    alpha = np.repeat(1. / len(y), len(y))
    u = UnitStump(x[:, 0], y * alpha)
    e = StumpError(x[:, 0], y * alpha, u)
    w = .5 * np.log((1. - e) / e)
    alpha[y < 0] *= np.exp(w)
    alpha[y > 0] *= np.exp(-w)
    alpha /= np.sum(alpha)
    ans.append((u, w))
    for i in np.arange(np.size(x, 1)):
        u = DummyStump(x[:, i], y * alpha)
        e = StumpError(x[:, i], y * alpha, u)
        w = .5 * np.log((1. - e) / e)
        alpha[(x[:, i ] < u) & (y > 0)] *= np.exp(w)
        alpha[(x[:, i ] > u) & (y < 0)] *= np.exp(w)
        alpha[(x[:, i ] < u) & (y < 0)] *= np.exp(-w)
        alpha[(x[:, i ] > u) & (y > 0)] *= np.exp(-w)
        alpha /= np.sum(alpha)
        ans.append((u, w))
    return ans


def estimate_boosting_stump_with_cross_validation(x, y, n, d):
    '''
    Depth of x to predict y
    '''
    np.random.seed(0)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    step = 1. * len(y) / n
    e = []
    sh = []
    for l in np.arange(d):
        print('Estimating depth %d' % (l+1))
        tmp = []
        for i in np.arange(n):
            myx = x.diff(l + 1).fillna(0.).values
            myy = y.fillna(0.).values
            x0 = np.int(step * i)
            x1 = len(y) if i == n-1 else np.int(step * (i+1))
            xs = np.vstack([myx[idx[:x0]], myx[idx[x1:]]])
            ys = np.array(list(myy[idx[:x0]]) + list(myy[idx[x1:]]))
            xv = myx[idx[x0:x1]]
            yv = myy[idx[x0:x1]]
            ans = BoostingStump(xs[:, :l+1], ys)
            tmp.append(StumpValidationError(xv, yv, ans))
        e.append(np.mean(tmp))
        sh.append((.5 - np.mean(tmp)) / np.std(tmp))
    lx = np.argsort(sh)[-1] + 1
    me = np.max(sh)
#     lx = None
#     me = 1e6
#     for i, ee in enumerate(e):
#         if ee <= me:
#             me = ee
#             lx = i + 1 
    ans = BoostingStump(x.diff(lx).fillna(0.).values, y.fillna(0.).values)
    return ans, lx, np.array([.5] + e), me


def greedy_boosting_stump(x, y, n):
    selected = []
    remaining = list(np.arange(np.size(x, 1)))
    error = 0.5
    final_ans = None
    final_lx = None
    final_e = None 
    while len(remaining)>0:
        tmp_error = error
        new_item = None
        for i in remaining:
            xx = x.iloc[:, selected + [i]]
            ans, lx, e, me = estimate_boosting_stump_with_cross_validation(xx, y, n, 12)
            if me < tmp_error:
                tmp_error = me
                new_item = i
                final_ans = ans
                final_lx = lx
                final_e = e
        if new_item is None:
            break
        else:
            selected.append(new_item)
            remaining = [i for i in remaining if i != new_item]
            error = tmp_error
    return final_ans, final_lx, final_e, error, selected


def BoostingLogistic(x, y, alpha, numIterations, epsilon, l):
    ans = []
    a = np.repeat(1. / len(y), len(y))
    for i in np.arange(len(x)):
        theta, _ = gradientAscent(x[i], y * a, np.zeros(np.size(x[0], 1)), alpha, numIterations, epsilon, l)
        pred = LogisticPrediction(x[i], theta)
        e = StumpError(pred, y * a, 0.)
        w = .5 * np.log((1. - e) / e)
        a[(pred < 0) & (y > 0)] *= np.exp(w)
        a[(pred > 0) & (y < 0)] *= np.exp(w)
        a[(pred < 0) & (y < 0)] *= np.exp(-w)
        a[(pred > 0) & (y > 0)] *= np.exp(-w)
        a /= np.sum(a)
        ans.append((theta, w))
    return ans


def estimate_boosting_logistic_with_cross_validation(x, y, n, d, alpha, numIterations, epsilon, l):
    np.random.seed(0)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    step = 1. * len(y) / n
    e = []
    for l in np.arange(d):
        print('Estimating depth %d' % (l+1))
        tmp = []
        for i in np.arange(n):
            myy = y.fillna(0.).values
            x0 = np.int(step * i)
            x1 = len(y) if i == n-1 else np.int(step * (i+1))
            xsid = list(idx[:x0]) + list(idx[x1:])
            xvid = list(idx[x0:x1])
            xs = [x[j].diff(l+1).fillna(0.).values[xsid] for j in xrange(len(x))]
            ys = myy[xsid]
            xv = [x[j].diff(l+1).fillna(0.).values[xvid] for j in xrange(len(x))]
            yv = myy[xvid]
            ans = BoostingLogistic(xs, ys, alpha, numIterations, epsilon, l)
            tmp.append(LogisticValidationError(xv, yv, ans))
        e.append(np.mean(tmp))
    lx = None
    me = 1e6
    for i, ee in enumerate(e):
        if ee <= me:
            me = ee
            lx = i + 1
    ans = BoostingLogistic([x[j].diff(lx).fillna(0.).values for j in xrange(len(x))], y.fillna(0.).values, alpha, numIterations, epsilon, l)
    return ans, lx, np.array([.5] + e), me


def plot_boosting_error(e):
    plt.figure()
    plt.bar(np.arange(len(e)), e, align='center')
    plt.axhline(0., color='black')
    plt.ylabel('Error')
    plt.xlabel('Depth')
    plt.xlim((-1, len(e)))

    
def test_econ(plot_boosting=False):
    raw = econ.read_bloomberg_data()
    release = econ.get_bloomberg_release(raw)
    change = econ.get_bloomberg_change(raw)
    change.columns = ['d' + x for x in change.columns]
    data = pd.concat([release, change], axis=1)
    p = read_price_data()
    pw = p['SPX Index'].dropna().resample('W', 'last')
    p = p['SPX Index'].dropna().resample('M', 'last')
    r = p.diff() / p.shift()
    r = r[dt(2000,1,1):].fillna(0.)
    data = tu.resample(data, pw)
    med = data.median(axis=0)
    data = pd.ewma(data, span=23)
    data = tu.resample(data, r)
    ans = BoostingStump(data.shift()[:dt(2017,2,12)].fillna(med).values, r[:dt(2017,2,12)].values)
    pred = np.sign(pd.Series(ValidationPrediction(data.shift().fillna(med).values, ans), index=data.index))
    w = pred * .5 + .5
    predx = np.sign(pd.Series(ValidationPrediction(data.fillna(med).values, ans), index=data.index))
    r1 = r
    r2 = r * w
    r3 = r2 - r1
    plt.figure()
    r1.cumsum().plot(label='Original %.2f' % (r1.mean() / r1.std() * np.sqrt(12.)))
    r2.cumsum().plot(label='New %.2f' % (r2.mean() / r2.std() * np.sqrt(12.)))
    r3.cumsum().plot(label='Diff %.2f' % (r3.mean() / r3.std() * np.sqrt(12.)))
    plt.legend(loc='best', frameon=False)
    return pred, predx, r


def test_econ_parameter():
    raw = econ.read_bloomberg_data()
    release = econ.get_bloomberg_release(raw)
    change = econ.get_bloomberg_change(raw)
    change.columns = ['d' + x for x in change.columns]
    data = pd.concat([release, change], axis=1)
    p = read_price_data()
    pw = p['SPX Index'].dropna().resample('W', 'last')
    p = p['SPX Index'].dropna().resample('M', 'last')
    r = p.diff() / p.shift()
    r = r[dt(2000,1,1):dt.today()].fillna(0.)
    data = tu.resample(data, pw)
    error = pd.Series([])
    for k in xrange(2, 26):
        med = data.median(axis=0)
        data1 = pd.ewma(data, span=k).shift().fillna(med)
        x = tu.resample(data1, r)
        ans = BoostingStump(x.values, r.fillna(0.).values)
        error.loc[k] = StumpValidationError(x.values, r.fillna(0.).values, ans)
    return error


def test_econ_weekly():
    raw = econ.read_bloomberg_data()
    release = econ.get_bloomberg_release(raw)
    change = econ.get_bloomberg_change(raw)
    change.columns = ['d' + x for x in change.columns]
    data = pd.concat([release, change], axis=1)
    p = read_price_data()
    p = p['SPX Index'].dropna()
    r = p.resample('W', 'last').ffill().diff()
    r = r[dt(2000,1,1):].fillna(0.)
    data = tu.resample(data, r)
    med = data.median(axis=0)
    data = pd.ewma(data, span=15)
    ans = BoostingStump(data.shift()[:dt(2017,2,12)].fillna(med).values, r[:dt(2017,2,12)].values)
    w = np.sign(pd.Series(ValidationPrediction(data.shift().fillna(med).values, ans), index=data.index))
    wp = np.sign(pd.Series(ValidationPrediction(data.fillna(med).values, ans), index=data.index))
    r1 = r
    r2 = r * w
    r3 = r2 - r1
    plt.figure()
    r1.cumsum().plot(label='Original %.2f' % (r1.mean() / r1.std() * np.sqrt(52.)))
    r2.cumsum().plot(label='New %.2f' % (r2.mean() / r2.std() * np.sqrt(52.)))
    r3.cumsum().plot(label='Diff %.2f' % (r3.mean() / r3.std() * np.sqrt(52.)))
    plt.legend(loc='best', frameon=False)
    return w, wp, r


def test_econ_weekly_parameter():
    raw = econ.read_bloomberg_data()
    release = econ.get_bloomberg_release(raw)
    change = econ.get_bloomberg_change(raw)
    change.columns = ['d' + x for x in change.columns]
    data = pd.concat([release, change], axis=1)
    p = read_price_data()
    p = p['SPX Index'].dropna()
    r = p.resample('W', 'last').ffill().diff()
    r = r[dt(2000,1,1):dt.today()]
    data = tu.resample(data, r)
    error = pd.Series([])
    for k in xrange(2, 26):
        med = data.median(axis=0)
        x = pd.ewma(data, span=k).shift().fillna(med)
        ans = BoostingStump(x.values, r.fillna(0.).values)
        error.loc[k] = StumpValidationError(x.values, r.fillna(0.).values, ans)
    return error


def test_fx_econ_weekly():
    raw = econ.read_bloomberg_data()
    release = econ.get_bloomberg_release(raw)
    change = econ.get_bloomberg_change(raw)
    change.columns = ['d' + x for x in change.columns]
    data = pd.concat([release, change], axis=1)
    p = read_price_data('FX')
    p = p['DXY Curncy'].dropna()
    r = p.resample('W', 'last').diff()
    r = r[dt(2000,1,1):].fillna(0.)
    data = tu.resample(data, r)
    med = data[:dt(2017,3,1)].median(axis=0)
    data = pd.ewma(data, span=21).fillna(med)
    ans = BoostingStump(data.shift()[:dt(2017,2,12)].fillna(med).values, r[:dt(2017,2,12)].values)
    w = np.sign(pd.Series(ValidationPrediction(data.shift().fillna(med).values, ans), index=data.index))
    wp = np.sign(pd.Series(ValidationPrediction(data.fillna(med).values, ans), index=data.index))
    r1 = r
    r2 = r * w
    r3 = r2 - r1
    plt.figure()
    r1.cumsum().plot(label='Original %.2f' % (r1.mean() / r1.std() * np.sqrt(52.)))
    r2.cumsum().plot(label='New %.2f' % (r2.mean() / r2.std() * np.sqrt(52.)))
    r3.cumsum().plot(label='Diff %.2f' % (r3.mean() / r3.std() * np.sqrt(52.)))
    plt.legend(loc='best', frameon=False)
    return w, wp, r


def test_fx_econ_weekly_parameter():
    raw = econ.read_bloomberg_data()
    release = econ.get_bloomberg_release(raw)
    change = econ.get_bloomberg_change(raw)
    change.columns = ['d' + x for x in change.columns]
    data = pd.concat([release, change], axis=1)
    p = read_price_data('FX')
    p = p['DXY Curncy'].dropna()
    r = p.resample('W', 'last').diff()
    r = r[dt(2000,1,1):dt.today()].fillna(0.)
    data = tu.resample(data, r)
    error = pd.Series([])
    for k in xrange(2, 26):
        med = data.median(axis=0)
        data1 = pd.ewma(data, span=k).shift().fillna(med)
        ans = BoostingStump(data1.values, r.values)
        error.loc[k] = StumpValidationError(data1.values, r.values, ans)
    return error


def get_econ_data():
    raw = econ.read_bloomberg_data()
    release = econ.get_bloomberg_release(raw)
    change = econ.get_bloomberg_change(raw)
    change.columns = ['d' + x for x in change.columns]
    data = pd.concat([release, change], axis=1)
    return data


def construct_model(r, data, max_depth=30, lookback=10, models=100, n=5):
    y = r.values[1:]
    if max_depth > np.size(data, 1):
        max_depth = np.size(data, 1)
    np.random.seed(0)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    step = 1. * len(y) / n
    trees = []
    tmp = np.arange(np.size(data, 1))
    for i in xrange(models):
        np.random.shuffle(tmp)
        trees.append(tmp.copy())
    trees = np.array(trees)
    max_improvement = 1.
    target_depth = 0
    target_lookback = 0
    for depth in xrange(max_depth):
        for l in xrange(1, lookback):
            x = pd.ewma(data, span=l+1).values[:-1]
            errors = []
            for i in np.arange(n):
                print('Estimating depth %d lookback %d bucket %d' % (depth + 1, l + 1, i + 1))
                x0 = np.int(step * i)
                x1 = len(y) if i == n-1 else np.int(step * (i+1))
                xs = np.vstack([x[idx[:x0]], x[idx[x1:]]])
                ys = np.array(list(y[idx[:x0]]) + list(y[idx[x1:]]))
                xv = x[idx[x0:x1]]
                yv = y[idx[x0:x1]]
                predictions = []
                for t in xrange(models):
                    branch = trees[t, :depth+1]
                    ans = BoostingStump(xs[:, branch], ys)
                    predictions.append(ValidationPrediction(xv[:, branch], ans))
                predictions = np.mean(np.array(predictions), axis=0)
                errors.append(StumpError(predictions, yv, 0))
            # im = (.5 - np.mean(errors)) / np.std(errors)
            im = np.mean(errors)
            if im < max_improvement:
                max_improvement = im
                target_depth = depth
                target_lookback = l
    print('Target Depth %d, Lookback: %d, Improvement %f' % (target_depth + 1, target_lookback + 1, max_improvement))
    x = pd.ewma(data, span=target_lookback+1).values[:-1]
    target_models = []
    for t in xrange(models):
        branch = trees[t, :target_depth+1]
        target_models.append(BoostingStump(x[:, branch], y))
    return trees, target_depth, target_lookback, target_models
            
    
def run_fx_econ(build_model=True):
    sample_date = dt(2017,1,1)
    p = read_price_data('FX')
    p = p['DXY Curncy'].dropna()
    r = p.resample('W', 'last').diff()
    r = r[dt(2000,1,1):].fillna(0.)
    data = get_econ_data()
    print('%d series' % np.size(data, 1))
    data = tu.resample(data, r)
    data = data.fillna(data[:sample_date].median())

    r_is = r[:sample_date]
    data_is = data[:sample_date]
    if build_model:
        print('Constructing model')
        trees, target_depth, target_lookback, target_models = construct_model(r_is, data_is)
        with open(PATH + 'FXECON.model', 'wb') as f:
            pickle.dump((trees, target_depth, target_lookback, target_models), f)
            f.close()
    else:
        print('Loading model')
        with open(PATH + 'FXECON.model', 'rb') as f:
            trees, target_depth, target_lookback, target_models = pickle.load(f)
            f.close()
    
    print('Target Depth %d, Lookback: %d' % (target_depth + 1, target_lookback + 1))
    x = pd.ewma(data, span=target_lookback+1)
    pred = []
    for t in xrange(len(trees)):
        branch = trees[t, :target_depth+1]
        pred.append(ValidationPrediction(x.values[:, branch], target_models[t]))
    pred = pd.Series(np.sign(np.mean(np.array(pred), axis=0)), index=data.index)
    r1 = r
    r2 = r * pred.shift()
    r3 = r2 - r1
    plt.figure()
    r1.cumsum().plot(label='Original %.2f' % (r1.mean() / r1.std() * np.sqrt(52.)))
    r2.cumsum().plot(label='New %.2f' % (r2.mean() / r2.std() * np.sqrt(52.)))
    r3.cumsum().plot(label='Diff %.2f' % (r3.mean() / r3.std() * np.sqrt(52.)))
    plt.legend(loc='best', frameon=False)
    return pred.shift(), pred, r
    

def run_pension():
    p1, px, r = test_econ()
    pnl = (.5 * p1 + .5) * r
    px.name = 'Bloomberg Pred'
    pnl.name = 'Bloomberg'
    print(pd.concat([pnl, px], axis=1).iloc[-10:])
    a = pd.Series([pnl.mean(), pnl.std(), pnl.mean() / pnl.std() * np.sqrt(12.), _five_percent(pnl), _one_percent(pnl), pnl.min()],
                  index = ['Mean', 'Std', 'Sharpe', '5%', '1%', 'Worst'])
    print(a)


def _one_percent(x):
    return np.percentile(x.dropna().values, 1)


def _five_percent(x):
    return np.percentile(x.dropna().values, 5)


def run_trade():
    p1, pp, r = test_econ_weekly()
    pnl = p1 * r
    pp.name = 'Prediction'
    pnl.name = 'PnL'
    print(pd.concat([pnl, pp], axis=1).iloc[-10:])
    a = pd.Series([pnl.mean(), pnl.std(), pnl.mean() / pnl.std() * np.sqrt(52.), _five_percent(pnl), _one_percent(pnl), pnl.min()],
                  index = ['Mean', 'Std', 'Sharpe', '5%', '1%', 'Worst'])
    print(a)


def run_trade_fx(build_model=False):
    p1, pp, r = test_fx_econ_weekly()
    # p1, pp, r = run_fx_econ(build_model)
    pnl = p1 * r
    pp.name = 'Prediction'
    pnl.name = 'PnL'
    print(pd.concat([pnl, pp], axis=1).iloc[-10:])
    a = pd.Series([pnl.mean(), pnl.std(), pnl.mean() / pnl.std() * np.sqrt(52.), _five_percent(pnl), _one_percent(pnl), pnl.min()],
                  index = ['Mean', 'Std', 'Sharpe', '5%', '1%', 'Worst'])
    print(a)


if __name__ == '__main__':
    run_trade()
