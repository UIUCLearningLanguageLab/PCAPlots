import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src import config


def make_scree_fig():
    """
    Returns fig showing amount of variance accounted for by each principal component
    """

    # load data
    acts_mat = model.term_acts_mat if config.Fig.SVD_IS_TERMS else model.get_multi_probe_prototype_acts_df().values
    pca_model = PCA(n_components=acts_mat.shape[1])
    pca_model.fit(acts_mat)
    expl_var_perc = np.asarray(pca_model.explained_variance_ratio_) * 100

    # fig
    fig, ax = plt.subplots(figsize=(config.Fig.fig_size, 3), dpi=config.Fig.dpi)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.set_xticklabels([])
    ax.set_xlabel('Principal Component', fontsize=config.Fig.ax_label_fontsize)
    ax.set_ylabel('% Var Explained (Cumulative)', fontsize=config.Fig.ax_label_fontsize)
    ax.set_ylim([0, 100])

    # plot
    ax.plot(expl_var_perc.cumsum(), '-', linewidth=config.Fig.LINEWIDTH, color='black')
    ax.plot(expl_var_perc.cumsum()[:config.Fig.NUM_PCS], 'o', linewidth=config.Fig.LINEWIDTH, color='black')
    fig.tight_layout()

    return fig
