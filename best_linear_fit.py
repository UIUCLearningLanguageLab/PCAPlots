import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src import config
from src.utils import plot_best_fit_line


def make_linear_fit_fig(x: np.ndarray,
                        y: np.ndarray,
                        ) -> plt.Figure:
    """
    Returns fig showing scatter and best linear fit line
    """

    # fig
    res, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Var 1', fontsize=config.Fig.ax_label_fontsize)
    ax.set_ylabel('Var 2', fontsize=config.Fig.ax_label_fontsize)

    # plot
    ax.scatter(x, y)
    plot_best_fit_line(ax, x, y, fontsize=16)

    return res


# create random data
var1 = np.array([0, 0, 1, 3, 4, 2, 3, 6, 5, 4, 7, 6, 8, 7, 9, 8])
var2 = np.roll(var1, shift=-1)

fig = make_linear_fit_fig(var1, var2)
fig.show()
