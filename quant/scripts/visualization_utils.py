'''
Created on Oct 5, 2014

@author: Wayne
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from lib import timeseries_utils as tu

COLORMAP = ['#1E90FF', '#FEEB00', '#FFC214', '#9CCD71', '#1AB55A', '#FF7076', '#FF4853', '#9595E4']


def bar_plot(data, title=None):
    if isinstance(data, pd.Series):
        data = pd.DataFrame(np.array([data.values]), columns=data.index, index=[data.name])
    data = data.fillna(0.)
    n, k = np.shape(data)
    if n>1:
        recs = []
        for i in range(n):
            plt.bar(np.arange(k) * (n+1) + i, data.iloc[i], align='center', color=COLORMAP[i], edgecolor=COLORMAP[i])
            recs.append(plt.Rectangle((0,0),1,1, color=COLORMAP[i]))
        plt.xticks(np.arange(k) * (n+1) + n/2, data.columns)
        plt.xlim((-1, k * (n+1)))
        plt.legend(recs, list(data.index), frameon=False, loc='best', prop={'size': 9})
    else:
        plt.bar(np.arange(k), data.iloc[0], align='center', color=COLORMAP[0], edgecolor=COLORMAP[0])
        plt.xticks(np.arange(k), data.columns)
        plt.xlim((-1, k))
    plt.axhline(0., color='grey')
    if title is not None:
        plt.title(title, weight='bold')


def axis_area_plot(ts, title=None, color=COLORMAP[0]):
    ts = tu.ts_interpolate(ts)
    plt.fill_between(ts[ts>=0].index, np.zeros(len(ts[ts>=0])), ts[ts>=0].values, color=color)
    plt.fill_between(ts[ts<=0].index, np.zeros(len(ts[ts<=0])), ts[ts<=0].values, color=color)
    if title is not None:
        plt.title(title, weight='bold')


def bin_plot(x, y, bins=20, diag_line = False):
    if (isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)) and (isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)):
        data = pd.concat([x, y], axis=1)
    else:
        x = pd.Series(np.array(x).flatten())
        y = pd.Series(np.array(y).flatten())
        data = pd.concat([x, y], axis=1)
    data = data.ix[:,:2].dropna()
    assert len(data) > bins
    tmp_data = np.array(sorted([tuple(v) for v in data.values]))
    step_size = 1. * len(tmp_data) / bins
    ym = []
    xm = []
    for i in range(bins):
        start = np.int(np.round(i * step_size))
        end = np.int(np.round((i+1) * step_size))
        if end > len(tmp_data):
            end = len(tmp_data)
        tmp = tmp_data[start:end]
        xm.append(np.mean(tmp[:,0]))
        ym.append(np.mean(tmp[:,1]))
        plt.fill_between(np.linspace(np.min(tmp[:,0]), np.max(tmp[:,0])), np.min(tmp[:,1]), np.max(tmp[:,1]),
                         color='green', alpha=0.5)
    plt.scatter(xm, ym, color='green', marker='o', s=20)
    if diag_line:
        dl = np.linspace(np.min(xm), np.max(xm))
        plt.plot(dl, dl, ls='-', lw=1, color='blue', marker=None)
    if data.icol(0).min() * data.icol(0).max() < 0:
        plt.axvline(0., color='red')
    if data.icol(1).min() * data.icol(1).max() < 0:
        plt.axhline(0., color='red')


def winners_plot(prices, winners):
    winners = winners.dropna()
    prices = tu.resample(prices, winners)
    p = pd.concat([prices[['Open', 'Close']].min(axis=1), prices[['Open', 'Close']].max(axis=1)], axis=1)
    pos = winners[winners>=0]
    neg = winners[winners<0]
    [plt.bar(pos.index[i], p.loc[pos.index[i]].values[1] - p.loc[pos.index[i]].values[0], 0.9,
             p.loc[pos.index[i]].values[0], color='green', align='center', lw=0, alpha=0.3 + 0.7 * pos.values[i]/pos.max())
     for i in range(len(pos))]
    [plt.plot((pos.index[i], pos.index[i]), (prices['Low'].loc[pos.index[i]], prices['High'].loc[pos.index[i]]),
              color='green', alpha=0.3 + 0.7 * pos.values[i]/pos.max()) for i in range(len(pos))]
    [plt.bar(neg.index[i], p.loc[neg.index[i]].values[1] - p.loc[neg.index[i]].values[0], 0.9,
             p.loc[neg.index[i]].values[0], color='red', align='center', lw=0, alpha=0.3 + 0.7 * neg.values[i]/neg.min())
     for i in range(len(neg))]
    [plt.plot((neg.index[i], neg.index[i]), (prices['Low'].loc[neg.index[i]], prices['High'].loc[neg.index[i]]),
              color='red', alpha=0.3 + 0.7 * neg.values[i]/neg.min()) for i in range(len(neg))]