import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import argparse
import pandas as pd
import netCDF4 
import pprint as pprint
    # Import libraries
from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import os


def sigmoid(x, L ,x0, k, b):

    y = L / (1 + np.exp(-k*(x-x0)))+b

    return (y)

    
def get_joint_histgram(nx, ny, xmin, xmax, ymin, ymax, x, y):
    """
    :param nx: num of x bins
    :param ny: num of y bins
    :param xmin: lower bound of x bins
    :param xmax: higher bound of x bins
    :param ymin: lower bound of y bins
    :param ymax: higher bound Eof y bins
    :param x: 1D data of x
    :param y: 1D data of y
    :return: jh(ny, nx), x_mid(nx), y_mid(ny)
    """
    jh = np.empty([ny, nx])
    y_median = np.empty([nx])
    x_bins = np.linspace(xmin,xmax, nx+1) #bin_edges
    y_bins = np.linspace(ymin,ymax, ny+1)
    x_mid  = x_bins[:-1] + (x_bins[1] - x_bins[0]) / 2
    y_mid  = y_bins[:-1] + (y_bins[1] - y_bins[0]) / 2
    # print(x_bins)
    # print(y_bins)
    labels = ['Bin {}'.format(i) for i in range(1, nx+1)]
    x_bin = pd.cut(x, x_bins, labels = labels) #x_bin match both x and y
    i = 0
    for bin in labels:
        # print(df.AI[df.x_bins == bin].mean(), df.Nd[df.AI_bins == bin].mean())
        y_median[i] =  y[x_bin == bin].median()
        dict_jh = get_PDF_bin_range(y[x_bin == bin], y_bins)
        jh[:, i] = dict_jh['pdf']
        i += 1
    # fig, ax = plt.subplots()
    # cs = ax.contourf(x_mid, y_mid, jh, cmap='Greys')
    # fig.colorbar(cs, ax=ax)
    return (jh, x_mid, y_mid, y_median)

def get_PDF_bin_range(x, x_bins):
    """
    :param x: 1-D array data
    :param x_bins: bin-edges (length = n_bins+1)
    :return: dict: 1) 'x': mid_value of x for each bin
                   2) 'pdf': PDF for each bin (%)
    """
    dict_PDF        ={}
    hist, bin_edges = np.histogram(x, bins=x_bins)
    dict_PDF['x']   = bin_edges[0:len(x_bins)-1]+(bin_edges[1]-bin_edges[0])/2 #mid value
    dict_PDF['pdf'] = hist/sum(hist)*100
    # fig, ax = plt.subplots()
    # ax.plot(dict_PDF['x'], dict_PDF['pdf'], color='black', linewidth=3)
    return (dict_PDF)    


def get_values_joint_hist(xedges, yedges):
    """
    :param nx: num of x bins  20
    :param ny: num of y bins  20 
    :param xmin: lower bound of x bins 
    :param xmax: higher bound of x bins
    :param ymin: lower bound of y bins
    :param ymax: higher bound Eof y bins
    :param x: 1D data of x
    :param y: 1D data of y
    :return: jh(ny, nx), x_mid(nx), y_mid(ny)
    """
    
    xedges_mid  = xedges[:-1] + (xedges[1] - xedges[0]) / 2
    yedges_mid  = yedges[:-1] + (yedges[1] - yedges[0]) / 2
    print("========print xedges_mid", xedges_mid)
    print("========print yedges_mid", yedges_mid)

    return (xedges_mid, yedges_mid)



