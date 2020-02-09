import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects
from math import gcd

from src import config


def equidistant_elements(l, n):
    """return "n" elements from "l" such that they are equally far apart iin "l" """
    while not gcd(n, len(l)) == n:
        l.pop()
    step = len(l) // n
    ids = np.arange(step, len(l) + step, step) - 1  # -1 for indexing
    res = np.asarray(l)[ids].tolist()
    return res


def make_cat_acts_2d_walk_fig(embeddings: np.ndarray,
                              component1: int,
                              component2: int,
                              num_ticks: int,
                              ) -> plt.Figure:
    """
    Returns fig showing evolution of embeddings in 2D space using PCA.
    """

    assert np.ndim(embeddings) == 3  # (ticks, words, embedding dimensions)

    palette = np.array(sns.color_palette("hls", embeddings.shape[1]))
    model_ticks = [n for n, _ in enumerate(embeddings)]
    equidistant_ticks = equidistant_elements(model_ticks, num_ticks)

    # fit pca model on last tick
    num_components = component2
    pca_model = PCA(n_components=num_components)
    pca_model.fit(embeddings[-1])

    # transform embeddings at requested ticks with pca model
    transformations = []
    for ei in embeddings[equidistant_ticks]:
        transformations.append(pca_model.transform(ei)[:, [component1, component2]])

    # fig
    fig, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    for cat_id, cat in enumerate(model.hub.probe_store.cats):
        x, y = zip(*[acts_2d_mat[cat_id] for acts_2d_mat in transformations])
        ax.plot(x, y, c=palette[cat_id], lw=config.Fig.LINEWIDTH)
        xtext, ytext = transformations[-1][cat_id, :]
        txt = ax.text(xtext, ytext, str(cat), fontsize=8,
                      color=palette[cat_id])
        txt.set_path_effects([
            patheffects.Stroke(linewidth=config.Fig.LINEWIDTH, foreground="w"), patheffects.Normal()])
    ax.axis('off')
    x_max = np.max(np.dstack(transformations)[:, 0, :]) * 1.2
    y_max = np.max(np.dstack(transformations)[:, 1, :]) * 1.2
    ax.set_xlim([-x_max, x_max])
    ax.set_ylim([-y_max, y_max])
    ax.axhline(y=0, linestyle='--', c='grey', linewidth=1.0)
    ax.axvline(x=0, linestyle='--', c='grey', linewidth=1.0)
    ax.set_title(f'Principal components {component1} and {component2}\nEvolution across training')
    fig.tight_layout()

    return fig
