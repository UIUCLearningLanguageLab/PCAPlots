from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

from src import config


def make_dendrogram_heatmap_barplot_fig(similarity_matrix: np.ndarray,
                                        labels: List[str],
                                        frequencies: List[int],
                                        num_colors=None,
                                        vmin: float = 0.0,
                                        vmax: float = 1.0,
                                        cluster=False):
    """
    Returns fig showing dendrogram heatmap of similarity matrix and a bar plot
    """

    assert len(labels) == len(similarity_matrix)

    print('Probe simmat  min: {} max {}'.format(np.min(similarity_matrix), np.max(similarity_matrix)))
    print('Fig  min: {} max {}'.format(vmin, vmax))

    # fig
    res, ax_heatmap = plt.subplots(figsize=(11, 7), dpi=config.Fig.dpi)
    ax_heatmap.yaxis.tick_right()
    divider = make_axes_locatable(ax_heatmap)
    ax_dend = divider.append_axes("bottom", 0.8, pad=0.0)  # , sharex=ax_heatmap)
    ax_colorbar = divider.append_axes("left", 0.1, pad=0.4)
    ax_freqs = divider.append_axes("right", 4.0, pad=0.0)  # , sharey=ax_heatmap)

    # dendrogram
    ax_dend.set_frame_on(False)
    lnk0 = linkage(pdist(similarity_matrix))
    if num_colors is None or num_colors <= 1:
        left_threshold = -1
    else:
        left_threshold = 0.5 * (lnk0[1 - num_colors, 2] +
                                lnk0[-num_colors, 2])
    dg0 = dendrogram(lnk0, ax=ax_dend,
                     orientation='bottom',
                     color_threshold=left_threshold,
                     no_labels=True,
                     no_plot=not cluster)
    if cluster:
        # Reorder the values in x to match the order of the leaves of the dendrograms
        z = similarity_matrix[dg0['leaves'], :]  # sorting rows
        z = z[:, dg0['leaves']]  # sorting columns for symmetry
        simmat_labels = np.array(labels)[dg0['leaves']]
    else:
        z = similarity_matrix
        simmat_labels = labels

    # probe freq bar plot
    y = range(len(labels))
    ax_freqs.barh(y, frequencies, color='black')
    ax_freqs.set_xlabel('Freq')
    max_frequency = max(frequencies)
    ax_freqs.set_xlim([0, max_frequency])
    ax_freqs.set_xticks([0, max_frequency])
    ax_freqs.set_xticklabels([0, max_frequency])
    ax_freq_lim0 = ax_freqs.get_ylim()[0] + 0.5
    ax_freq_lim1 = ax_freqs.get_ylim()[1] - 0.5
    ax_freqs.set_ylim([ax_freq_lim0, ax_freq_lim1])  # shift ticks to match heatmap
    ax_freqs.yaxis.set_ticks(y)
    ax_freqs.yaxis.set_ticklabels(simmat_labels, color='white')

    # heatmap
    max_extent = ax_dend.get_xlim()[1]
    im = ax_heatmap.imshow(z[::-1], aspect='auto',
                           cmap=plt.cm.jet,
                           extent=(0, max_extent, 0, max_extent),
                           vmin=vmin, vmax=vmax)

    # colorbar
    cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='vertical')
    cb.ax.set_xticklabels([vmin, vmax], fontsize=config.Fig.ax_label_fontsize, rotation=90)
    cb.set_label('Correlation Coefficient', labelpad=-10, fontsize=config.Fig.ax_label_fontsize, rotation=90)

    # set heatmap tick labels
    ax_heatmap.xaxis.set_ticks([])
    ax_heatmap.xaxis.set_ticklabels([])
    ax_heatmap.yaxis.set_ticks([])
    ax_heatmap.yaxis.set_ticklabels([])

    # Hide all tick lines
    lines = (ax_heatmap.xaxis.get_ticklines() +
             ax_heatmap.yaxis.get_ticklines() +
             ax_dend.xaxis.get_ticklines() +
             ax_dend.yaxis.get_ticklines())
    plt.setp(lines, visible=False)

    # set label rotation and fontsize
    x_labels = ax_heatmap.xaxis.get_ticklabels()
    plt.setp(x_labels, rotation=-90)
    plt.setp(x_labels, fontsize=config.Fig.ax_label_fontsize)
    y_labels = ax_heatmap.yaxis.get_ticklabels()
    plt.setp(y_labels, rotation=0)
    plt.setp(y_labels, fontsize=config.Fig.ax_label_fontsize)

    # make dendrogram labels invisible
    plt.setp(ax_dend.get_yticklabels() + ax_dend.get_xticklabels(),
             visible=False)
    res.tight_layout()

    return res


NUM_WORDS = 12
NOISE = 0.3

# create random words and similarity matrix
words = [f'word-{n}' for n in range(NUM_WORDS)]
frequencies = [random.randint(0, 50) for _ in words]
tmp1 = np.random.random((1, NUM_WORDS)).repeat(NUM_WORDS//2, axis=0) + NOISE * np.random.random((NUM_WORDS//2, NUM_WORDS))
tmp2 = np.random.random((1, NUM_WORDS)).repeat(NUM_WORDS//2, axis=0) + NOISE * np.random.random((NUM_WORDS//2, NUM_WORDS))
sim_matrix = np.vstack([tmp1, tmp2])

fig = make_dendrogram_heatmap_barplot_fig(sim_matrix, words, frequencies)
fig.show()
