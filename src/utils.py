import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

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


def add_double_legend(ax, lines_list, labels, model_descs, y_offset=-0.2):  # requires figure height = 6
    box = ax.get_position()
    num_model_groups = len(model_descs)
    shrink_prop = 0.1 * num_model_groups  # TODO this doesn't always work well
    ax.set_position([box.x0, box.y0 + box.height * shrink_prop,  # Shrink vertically to make room for legend
                     box.width, box.height * (1 - shrink_prop)])
    leg1 = plt.legend([l[0] for l in lines_list], model_descs, loc='upper center',
                      bbox_to_anchor=(0.5, y_offset), ncol=2, frameon=False, fontsize=config.Fig.legend_fontsize)
    for lines in lines_list:
        for line in lines:
            line.set_color('black')
    plt.legend(lines_list[0], labels, loc='upper center',
               bbox_to_anchor=(0.5, y_offset + 0.1), ncol=3, frameon=False, fontsize=config.Fig.legend_fontsize)
    plt.gca().add_artist(leg1)  # order of legend creation matters here


def add_single_legend(ax, model_descs, y_offset=-0.25):
    box = ax.get_position()
    shrink_prop = 0.1 * len(model_descs)
    ax.set_position([box.x0, box.y0 + box.height * shrink_prop,  # Shrink vertically to make room for legend
                     box.width, box.height * (1 - shrink_prop)])
    plt.legend(loc='center', fontsize=config.Fig.legend_fontsize,
               bbox_to_anchor=(0.5, y_offset), ncol=2, frameon=False)