from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src import config


def make_dendrogram_heatmap_fig(similarity_matrix: np.ndarray,
                                labels: List[str],
                                num_colors=None,
                                y_title=False,
                                vmin=0.0,
                                vmax=1.0):
    """
    Returns fig showing dendrogram heatmap of similarity matrix
    """

    assert len(labels) == len(similarity_matrix)

    print('Matrix  min: {} max {}'.format(np.min(similarity_matrix), np.max(similarity_matrix)))
    print('Figure  min: {} max {}'.format(vmin, vmax))

    # fig
    res, ax_heatmap = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax_heatmap.yaxis.tick_right()
    divider = make_axes_locatable(ax_heatmap)
    ax_dendrogram_right = divider.append_axes("right", 0.8, pad=0.0, sharey=ax_heatmap)
    ax_dendrogram_right.set_frame_on(False)
    ax_colorbar = divider.append_axes("top", 0.1, pad=0.4)

    # dendrogram
    lnk0 = linkage(pdist(similarity_matrix))
    if num_colors is None or num_colors <= 1:
        left_threshold = -1
    else:
        left_threshold = 0.5 * (lnk0[1 - num_colors, 2] +
                                lnk0[-num_colors, 2])
    dg0 = dendrogram(lnk0, ax=ax_dendrogram_right,
                     orientation='right',
                     color_threshold=left_threshold,
                     no_labels=True)

    # Reorder the values in x to match the order of the leaves of the dendrograms
    z = similarity_matrix[dg0['leaves'], :]  # sorting rows
    z = z[:, dg0['leaves']]  # sorting columns for symmetry

    # heatmap
    max_extent = ax_dendrogram_right.get_ylim()[1]
    im = ax_heatmap.imshow(z[::-1], aspect='auto',
                           cmap=plt.cm.jet,
                           extent=(0, max_extent, 0, max_extent),
                           vmin=vmin, vmax=vmax)

    # colorbar
    cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='horizontal')
    cb.ax.set_xticklabels([vmin, vmax], fontsize=config.Fig.ax_label_fontsize)
    cb.set_label('Correlation Coefficient', labelpad=-10, fontsize=config.Fig.ax_label_fontsize)

    # set heatmap ticklabels
    xlim = ax_heatmap.get_xlim()[1]
    ncols = len(labels)
    halfxw = 0.5 * xlim / ncols
    ax_heatmap.xaxis.set_ticks(np.linspace(halfxw, xlim - halfxw, ncols))
    ax_heatmap.xaxis.set_ticklabels(np.array(labels)[dg0['leaves']])  # for symmetry
    ylim = ax_heatmap.get_ylim()[1]
    nrows = len(labels)
    halfyw = 0.5 * ylim / nrows
    if y_title:
        ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, nrows))
        ax_heatmap.yaxis.set_ticklabels(np.array(labels)[dg0['leaves']])

    # Hide all tick lines
    lines = (ax_heatmap.xaxis.get_ticklines() +
             ax_heatmap.yaxis.get_ticklines() +
             ax_dendrogram_right.xaxis.get_ticklines() +
             ax_dendrogram_right.yaxis.get_ticklines())
    plt.setp(lines, visible=False)

    # set label rotation and fontsize
    x_labels = ax_heatmap.xaxis.get_ticklabels()
    plt.setp(x_labels, rotation=-90)
    plt.setp(x_labels, fontsize=config.Fig.ax_label_fontsize)
    y_labels = ax_heatmap.yaxis.get_ticklabels()
    plt.setp(y_labels, rotation=0)
    plt.setp(y_labels, fontsize=config.Fig.ax_label_fontsize)

    # make dendrogram labels invisible
    plt.setp(ax_dendrogram_right.get_yticklabels() + ax_dendrogram_right.get_xticklabels(),
             visible=False)
    res.subplots_adjust(bottom=0.2)  # make room for tick labels
    res.tight_layout()

    return res


NUM_WORDS = 12
NOISE = 0.3

# create random words and random embeddings
words = [f'word-{n}' for n in range(NUM_WORDS)]
tmp1 = np.random.random((1, NUM_WORDS)).repeat(NUM_WORDS//2, axis=0) + NOISE * np.random.random((NUM_WORDS//2, NUM_WORDS))
tmp2 = np.random.random((1, NUM_WORDS)).repeat(NUM_WORDS//2, axis=0) + NOISE * np.random.random((NUM_WORDS//2, NUM_WORDS))
sim_matrix = np.vstack([tmp1, tmp2])

fig = make_dendrogram_heatmap_fig(sim_matrix, words)
fig.show()
