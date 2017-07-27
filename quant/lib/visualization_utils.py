'''
Created on 28 Jun 2017

@author: wayne
'''
import numpy as np
import pandas as pd
from datetime import datetime as dt
from matplotlib import pyplot as plt
from quant.lib import timeseries_utils as tu
from quant.lib.main_utils import logger

COLORMAP = ['#1E90FF', '#FEEB00', '#FFC214', '#9CCD71', '#1AB55A', '#FF7076', '#FF4853', '#9595E4']


def get_series_for_plot(data, default_name=''):
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    else:
        d = np.array(data)
        if d.ndim > 1:
            d = d[:, 0]
        return pd.Series(d, name=default_name)

    
def bin_plot(x, y, bins=20, diag_line=False):
    data = pd.concat([get_series_for_plot(x, 'x'), get_series_for_plot(y, 'y')], axis=1).dropna()
    if len(data) > bins:
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
        identify = data.min() * data.max()
        if identify.values[0] < 0:
            plt.axvline(0., color='red')
        if identify.values[1] < 0:
            plt.axhline(0., color='red')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
    else:
        logger.warn('Not enough data for bin plot')


def axis_area_plot(ts, color=COLORMAP[0]):
    if isinstance(ts, pd.DataFrame):
        data = ts.iloc[:, 0]
    else:
        data = ts
    data = tu.ts_interpolate(data)
    plt.fill_between(data[data>=0].index, np.zeros(len(data[data>=0])), data[data>=0].values, color=color, alpha=0.5)
    plt.fill_between(data[data<=0].index, np.zeros(len(data[data<=0])), data[data<=0].values, color=color, alpha=0.5)
    data.plot(lw=2, color=color)


def use_monthly_ticks(data):
    idx = data.resample('B').last().index
    idx = [idx[i + 1] for i in xrange(len(idx)-1) if idx[i].month != idx[i+1].month]
    if len(idx) > 0:
        txt = [dt.strftime(x, '%b\n%Y') if x.month==1 else dt.strftime(x, '%b') for x in idx]
        plt.xticks(idx, txt, rotation=0, ha='center')


def highlight_last_observation(data, color='green'):
    s = data if isinstance(data, pd.Series) else data.iloc[:, 0]
    plt.plot([data.index[-1]], [data.values[-1]], marker='s', color=color)