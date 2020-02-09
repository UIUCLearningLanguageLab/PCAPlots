from typing import List, Dict
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import paired_cosine_distances

from src import config


def make_comparison_fig(group2embedding_matrices: Dict[str, List[np.ndarray]],
                        vmin: float = 0.0,
                        ) -> plt.Figure:
    """
    Returns fig showing similarity matrix of probe similarity matrices of multiple models
    """

    group_names = sorted(group2embedding_matrices.keys())

    # get a flat list of embedding matrices, preserving group order
    embedding_matrices_flat = []
    group_names_flat = []
    for k, v in sorted(g2embedding_matrices.items()):
        embedding_matrices_flat.extend(v)
        group_names_flat.extend([k] * len(v))

    avg_sims = np.zeros((len(embedding_matrices_flat), len(embedding_matrices_flat)))
    for i, embeddings_i in enumerate(embedding_matrices_flat):
        for j, embeddings_j in enumerate(embedding_matrices_flat):
            avg_sims[i, j] = 1 - paired_cosine_distances(embeddings_i, embeddings_j).mean()

    # fig
    fig, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    mask = np.zeros_like(avg_sims, dtype=np.bool)
    mask[np.triu_indices_from(mask, 1)] = True
    sns.heatmap(avg_sims, ax=ax, square=True, annot=False,
                annot_kws={"size": 5}, cbar_kws={"shrink": .5},
                vmin=vmin, vmax=1.0, cmap='jet')  # , mask=mask

    # colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([vmin, 1.0])
    cbar.set_ticklabels([str(vmin), '1.0'])
    cbar.set_label('Similarity between Semantic Spaces')

    # ax (needs to be below plot for axes to be labeled)
    ax.set_yticks(np.arange(len(group_names_flat)) + 0.5)
    ax.set_xticks(np.arange(len(group_names_flat)) + 0.5)
    ax.set_yticklabels(group_names_flat, rotation=0)
    ax.set_xticklabels(group_names_flat, rotation=90)
    plt.tight_layout()

    return fig


NUM_WORDS = 12
EMBED_SIZE = 8
NUM_IN_GROUP = 4
NOISE = 0.9

# create random embeddings
tmp1 = np.random.random((1, NUM_WORDS)).repeat(NUM_WORDS // 2, axis=0)
tmp2 = np.random.random((1, NUM_WORDS)).repeat(NUM_WORDS // 2, axis=0)
group1 = [tmp1 + NOISE * np.random.random((NUM_WORDS // 2, NUM_WORDS)) for _ in range(NUM_IN_GROUP)]
group2 = [tmp2 + NOISE * np.random.random((NUM_WORDS // 2, NUM_WORDS)) for _ in range(NUM_IN_GROUP)]

g2embedding_matrices = {'group-1': group1, 'group-2': group2}

fig = make_comparison_fig(g2embedding_matrices, vmin=0.8)
fig.show()