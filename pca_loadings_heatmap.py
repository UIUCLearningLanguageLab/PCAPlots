from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import string

from src import config


def make_pca_loadings_heatmap_fig(embeddings: np.ndarray,
                                  words: List[str],
                                  group2words: Dict[str, List[str]],
                                  num_components: int,
                                  label_explained_var: bool = True,
                                  ) -> plt.Figure:
    """
    Returns res showing heatmap of average loadings for words in each group of "group2words"
    """

    assert np.ndim(embeddings) == 2  # (words, embedding size)
    assert len(embeddings) == len(words)

    group_names = sorted(group2words.keys())
    num_groups = len(group_names)

    # do PCA
    pca_model = PCA(n_components=num_components)
    transformation = pca_model.fit_transform(embeddings)
    explained_var_percent = np.asarray(pca_model.explained_variance_ratio_) * 100

    # group loadings + average over all loadings in group
    avg_loadings = []
    for g in group_names:
        row_ids = [words.index(w) for w in group2words[g]]
        avg_loading = transformation[row_ids, :num_components].mean(axis=0)
        avg_loadings.append(avg_loading)
    grouped_loadings = np.vstack(avg_loadings)

    # fig
    res, ax_heatmap = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    divider = make_axes_locatable(ax_heatmap)
    ax_colorbar = divider.append_axes("right", 0.2, pad=0.1)

    # cluster rows
    lnk0 = linkage(pdist(grouped_loadings))
    dg0 = dendrogram(lnk0,
                     no_plot=True)
    z = grouped_loadings[dg0['leaves'], :]

    # heatmap
    max_extent = ax_heatmap.get_ylim()[1]
    vmin, vmax = round(np.min(z), 1), round(np.max(z), 1)
    im = ax_heatmap.imshow(z[::-1], aspect='auto',
                           cmap=plt.cm.jet,
                           extent=(0, max_extent, 0, max_extent),
                           vmin=vmin, vmax=vmax)

    # color bar
    cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='vertical')
    cb.ax.set_xticklabels([vmin, vmax], fontsize=config.Fig.ax_label_fontsize)
    cb.set_label('Average Loading', labelpad=-10, fontsize=config.Fig.ax_label_fontsize)

    # set heatmap tick labels
    xlim = ax_heatmap.get_xlim()[1]
    halfxw = 0.5 * xlim / config.Fig.NUM_PCS
    ax_heatmap.xaxis.set_ticks(np.linspace(halfxw, xlim - halfxw, config.Fig.NUM_PCS))
    if label_explained_var:
        ax_heatmap.xaxis.set_ticklabels(['PC {} ({:.1f} %)'.format(pc_id + 1, expl_var)
                                         for pc_id, expl_var in zip(range(config.Fig.NUM_PCS), explained_var_percent)])
    else:
        ax_heatmap.xaxis.set_ticklabels(['PC {}'.format(pc_id + 1)
                                         for pc_id in range(config.Fig.NUM_PCS)])
    ylim = ax_heatmap.get_ylim()[1]
    halfyw = 0.5 * ylim / num_groups
    ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, num_groups))
    ax_heatmap.yaxis.set_ticklabels(np.array(group_names)[dg0['leaves']])

    # Hide all tick lines
    lines = (ax_heatmap.xaxis.get_ticklines() +
             ax_heatmap.yaxis.get_ticklines())
    plt.setp(lines, visible=False)

    # set label rotation and fontsize
    x_labels = ax_heatmap.xaxis.get_ticklabels()
    plt.setp(x_labels, rotation=-90)
    plt.setp(x_labels, fontsize=config.Fig.ax_label_fontsize)
    y_labels = ax_heatmap.yaxis.get_ticklabels()
    plt.setp(y_labels, rotation=0)
    plt.setp(y_labels, fontsize=config.Fig.ax_label_fontsize)

    return res


NUM_WORDS = 12
EMBED_SIZE = 8

# create random words and random embeddings
words = [f'word-{n}' for n in range(NUM_WORDS)]
g2words = {'group-1': words[:NUM_WORDS//2], 'group-2': words[NUM_WORDS//2:]}

embeddings = np.random.random((NUM_WORDS, EMBED_SIZE))

fig = make_pca_loadings_heatmap_fig(embeddings, words, g2words, num_components=EMBED_SIZE)
fig.show()