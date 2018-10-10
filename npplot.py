"""
This module contains functions for plotting.
"""
import numpy as np
import matplotlib.pylab as plt

def plot_np_fit_1d(x_t, y_t, x_data, y_data, y_star_mean, y_star_sigma, u_scale=1.0,
                   ylim=None, lloc="lower right", fignum=101):

    """
    Plots the observed data, posterior fit, and uncertainty of prediction.
    """
    fig = plt.figure(fignum)
    plt.plot(x_t, y_t, 'r:', label='$f(x)$')
    plt.plot(x_data[0:-2], y_data[0:-2], 'r.', markersize=10, label='Observations')
    plt.plot(x_t, y_star_mean, 'b-')

    plt.fill(np.concatenate([x_t, x_t[::-1]]),
             np.concatenate([y_star_mean - u_scale*1.9600 * y_star_sigma,
                             (y_star_mean + u_scale*1.9600 * y_star_sigma)[::-1]]), alpha=.5,
             fc='b', ec='None', label='Uncertainty')

    plt.plot(x_t[np.argmax(y_star_mean)], np.amax(y_star_mean), 'gx', markersize=20, label='Maximum')
    plt.plot(x_data[-2], y_data[-2], 'g.', markersize=20, label='Latest evaluation')
    plt.plot(x_data[-1], y_data[-1], 'k.', markersize=20, label='Next sample')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(ylim)
    plt.legend(loc=lloc)
    return fig
