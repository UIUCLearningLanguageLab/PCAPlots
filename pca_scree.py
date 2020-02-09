import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src import config


def make_scree_fig(embeddings: np.ndarray,
                   ) -> plt.Figure:
    """
    Returns fig showing amount of variance accounted for by each principal component
    """

    # do PCA
    pca_model = PCA(n_components=embeddings.shape[1])
    pca_model.fit(embeddings)
    explained_var_percent = np.asarray(pca_model.explained_variance_ratio_) * 100

    # fig
    fig, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_xlabel('Principal Component', fontsize=config.Fig.ax_label_fontsize)
    ax.set_ylabel('% Var Explained (Cumulative)', fontsize=config.Fig.ax_label_fontsize)
    ax.set_ylim([0, 100 + 1])

    # plot
    ax.plot(explained_var_percent.cumsum(), '-', linewidth=config.Fig.line_width, color='black')
    ax.plot(explained_var_percent.cumsum()[:config.Fig.NUM_PCS], 'o', linewidth=config.Fig.line_width, color='black')
    fig.tight_layout()

    return fig


NUM_WORDS = 12
EMBED_SIZE = 8

# create random embeddings
embeddings = np.random.random((NUM_WORDS, EMBED_SIZE))

fig = make_scree_fig(embeddings)
fig.show()
