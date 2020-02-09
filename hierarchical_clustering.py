import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src import config


def make_cat_cluster_fig(cat, bottom_off=False, max_probes=20, metric='cosine', x_max=config.Fig.CAT_CLUSTER_XLIM):
    """
    Returns fig showing hierarchical clustering of probes in single category
    """

    # load data
    cat_prototypes_df = model.get_single_cat_probe_prototype_acts_df(cat)
    if len(cat_prototypes_df) > max_probes:
        ids = np.random.choice(len(cat_prototypes_df) - 1, max_probes, replace=False)
        cat_prototypes_df = cat_prototypes_df.iloc[ids]
        probes_in_cat = cat_prototypes_df.index
    else:
        probes_in_cat = cat_prototypes_df.index.tolist()

    # fig
    fig, ax = plt.subplots(figsize=(config.Fig.fig_size, 4), dpi=config.Fig.dpi)
    dist_matrix = pdist(cat_prototypes_df.values, metric=metric)
    linkages = linkage(dist_matrix, method='complete')
    dendrogram(linkages,
               ax=ax,
               leaf_label_func=lambda x: probes_in_cat[x],
               orientation='right',
               leaf_font_size=8)
    ax.set_title(cat)
    ax.set_xlim([0, x_max])
    ax.tick_params(axis='both', which='both', top='off', right='off', left='off')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if bottom_off:
        ax.xaxis.set_ticklabels([])  # hides ticklabels
        ax.tick_params(axis='both', which='both', bottom='off')
        ax.spines['bottom'].set_visible(False)
    fig.tight_layout()

    return fig


def make_multi_cat_clust_fig(cats, metric='cosine'):
    """
    Returns fig showing hierarchical clustering of probes from multiple categories
    """

    # load data
    df = pd.DataFrame(pd.concat((model.get_single_cat_probe_prototype_acts_df(cat) for cat in cats), axis=0))
    cat_acts_mat = df.values
    cats_probe_list = df.index

    # fig
    fig, ax = plt.subplots(figsize=(config.Fig.fig_size, 5 * len(cats)), dpi=config.Fig.dpi)
    dist_matrix = pdist(cat_acts_mat, metric)
    linkages = linkage(dist_matrix, method='complete')
    dendrogram(linkages,
               ax=ax,
               labels=cats_probe_list,
               orientation='right',
               leaf_font_size=10)
    ax.tick_params(axis='both', which='both', top='off', right='off', left='off')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig
