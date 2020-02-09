import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects
from math import gcd
from typing import List
import string
import random

from src import config


def equidistant_elements(l, n):
    """return "n" elements from "l" such that they are equally far apart iin "l" """
    while not gcd(n, len(l)) == n:
        l.pop()
    step = len(l) // n
    ids = np.arange(step, len(l) + step, step) - 1  # -1 for indexing
    res = np.asarray(l)[ids].tolist()
    return res


def make_pca_across_time_fig(embeddings: np.ndarray,
                             words: List[str],
                             component1: int,
                             component2: int,
                             num_ticks: int,
                             ) -> plt.Figure:
    """
    Returns res showing evolution of embeddings in 2D space using PCA.
    """

    assert np.ndim(embeddings) == 3  # (ticks, words, embedding dimensions)
    assert len(words) == embeddings.shape[1]

    palette = np.array(sns.color_palette("hls", embeddings.shape[1]))
    model_ticks = [n for n, _ in enumerate(embeddings)]
    equidistant_ticks = equidistant_elements(model_ticks, num_ticks)

    # fit pca model on last tick
    num_components = component2 + 1
    pca_model = PCA(n_components=num_components)
    pca_model.fit(embeddings[-1])

    # transform embeddings at requested ticks with pca model
    transformations = []
    for ei in embeddings[equidistant_ticks]:
        transformations.append(pca_model.transform(ei)[:, [component1, component2]])

    # fig
    res, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.set_title(f'Principal components {component1} and {component2}\nEvolution across training')
    ax.axis('off')
    ax.axhline(y=0, linestyle='--', c='grey', linewidth=1.0)
    ax.axvline(x=0, linestyle='--', c='grey', linewidth=1.0)

    # plot
    for n, word in enumerate(words):

        # scatter
        x, y = zip(*[t[n] for t in transformations])
        ax.plot(x, y, c=palette[n], lw=config.Fig.LINEWIDTH)

        # text
        x_pos, y_pos = transformations[-1][n, :]
        txt = ax.text(x_pos, y_pos, str(word), fontsize=8,
                      color=palette[n])
        txt.set_path_effects([
            patheffects.Stroke(linewidth=config.Fig.LINEWIDTH, foreground="w"), patheffects.Normal()])

    return res


NUM_TICKS = 12
NUM_WORDS = 4
EMBED_SIZE = 8

# create random words and random embeddings
words = [f'word-{random.choice(string.ascii_letters)}' for _ in range(NUM_WORDS)]
embeddings = np.stack([np.random.random((NUM_WORDS, EMBED_SIZE)) * (NUM_TICKS / (tick + 1))
                       for tick in range(NUM_TICKS)])

fig = make_pca_across_time_fig(embeddings, words, component1=0, component2=1, num_ticks=6)
fig.show()