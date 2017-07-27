'''
Created on 28 Jun 2017

@author: wayne
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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


def axis_area_plot(ts, title=None, color=COLORMAP[0]):
    ts = tu.ts_interpolate(ts)
    plt.fill_between(ts[ts>=0].index, np.zeros(len(ts[ts>=0])), ts[ts>=0].values, color=color)
    plt.fill_between(ts[ts<=0].index, np.zeros(len(ts[ts<=0])), ts[ts<=0].values, color=color)
    if title is not None:
        plt.title(title, weight='bold')
