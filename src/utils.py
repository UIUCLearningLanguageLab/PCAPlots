import numpy as np
from scipy.stats import linregress

from src import config


def human_format(num, pos):  # pos is required for formatting mpl axis ticklabels
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format(num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def plot_best_fit_line(ax, x, y, fontsize, color='red', zorder=3, x_pos=0.05, y_pos=0.9, plot_p=True):

    # fit line
    try:
        best_fit_fxn = np.polyfit(x, y, 1, full=True)
    except Exception as e:  # cannot fit line
        print('Cannot fit line.', e)
        return

    # make line
    slope = best_fit_fxn[0][0]
    intercept = best_fit_fxn[0][1]
    xl = [min(x), max(x)]
    yl = [slope * xx + intercept for xx in xl]

    # plot line
    ax.plot(xl, yl, linewidth=config.Fig.line_width, c=color, zorder=zorder)

    # plot rsqrd
    variance = np.var(y)
    residuals = np.var([(slope * xx + intercept - yy) for xx, yy in zip(x, y)])
    Rsqr = np.round(1 - residuals / variance, decimals=3)
    if Rsqr > 0.5:
        fontsize += 5
    ax.text(x_pos, y_pos, '$R^2$ = {}'.format(Rsqr), transform=ax.transAxes, fontsize=fontsize)

    if plot_p:
        p = np.round(linregress(x, y)[3], decimals=8)
        ax.text(x_pos, y_pos - 0.05, 'p = {}'.format(p), transform=ax.transAxes, fontsize=fontsize - 2)
