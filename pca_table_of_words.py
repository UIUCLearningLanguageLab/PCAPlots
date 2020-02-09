from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import string

from src import config

plt.rcParams['font.family'] = 'monospace'  # each character in this font is equally wide


def make_principal_comps_table_fig(embeddings: np.ndarray,
                                   words: List[str],
                                   component: int,
                                   num_rows=4,
                                   w2f: Optional[Dict[str, int]] = None,
                                   fontsize: Optional[int] = None,
                                   ) -> plt.Figure:
    """
    Returns fig showing table of words loading highest and lowest on a principal component
    """

    assert np.ndim(embeddings) == 2  # (words, embedding size)
    assert len(embeddings) == len(words)

    if w2f is None:
        w2f = {w: 'n/a' for w in words}

    num_cols = 2
    col_labels = ['Low End', 'High End']

    # do PCA
    num_components = component + 1
    pca_model = PCA(n_components=num_components)
    transformation = pca_model.fit_transform(embeddings)
    explained_var_ratio = np.asarray(pca_model.explained_variance_ratio_) * 100
    pc = transformation[:, component]

    # sort and filter
    sorted_pc, sorted_words = list(zip(*sorted(zip(pc, words), key=lambda i: i[0])))

    col0_strings = [f'{w:<12} {loading:-2.2f} (freq: {w2f[w]})'
                    for w, loading in zip(sorted_words[:num_rows], sorted_pc[:num_rows])]

    col1_strings = [f'{w:<12} {loading:-2.2f} (freq: {w2f[w]})'
                    for w, loading in zip(sorted_words[-num_rows:][::-1], sorted_pc[-num_rows:][::-1])]

    # make matrix containing text
    max_rows = max(len(col0_strings), len(col1_strings))
    text_mat = np.chararray((max_rows, num_cols), itemsize=60, unicode=True)
    text_mat[:] = ''  # initialize so that mpl can read table
    text_mat[:len(col0_strings), 0] = col0_strings
    text_mat[:len(col1_strings), 1] = col1_strings

    # fig
    res, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.set_title('Principal Component {} ({:.2f}% var)'.format(
        component, explained_var_ratio[component]), fontsize=config.Fig.ax_label_fontsize)
    ax.axis('off')

    # plot table
    table_ = ax.table(cellText=text_mat, colLabels=col_labels, loc='center')
    if fontsize is not None:
        table_.auto_set_font_size(False)
        table_.set_fontsize(fontsize)
    res.tight_layout()

    return res


def make_loadings_line_fig(embeddings: np.ndarray,
                           component: int,
                           ) -> plt.Figure:
    """
    Returns fig showing line plot of loadings on a specified principal component
    """

    # do PCA
    num_components = component + 1
    pca_model = PCA(n_components=num_components)
    transformation = pca_model.fit_transform(embeddings)
    pc = transformation[:, component]

    # fig
    res, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plot
    ax.plot(sorted(pc), '-', linewidth=config.Fig.LINEWIDTH, color='black')
    ax.set_xticklabels([])
    ax.set_xlabel('Words', fontsize=config.Fig.ax_label_fontsize)
    ax.set_ylabel(f'Principal Component {component} Loading', fontsize=config.Fig.ax_label_fontsize)
    res.tight_layout()

    return res


NUM_WORDS = 12
EMBED_SIZE = 8

# create random words and random embeddings
words = [f'word-{random.choice(string.ascii_letters)}' for _ in range(NUM_WORDS)]
embeddings = np.random.random((NUM_WORDS, EMBED_SIZE))

fig = make_principal_comps_table_fig(embeddings, words, component=0, num_rows=6)
fig.show()

fig = make_loadings_line_fig(embeddings, component=0)
fig.show()