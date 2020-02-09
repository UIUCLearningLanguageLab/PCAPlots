from typing import List
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest

from src import config

plt.rcParams['font.family'] = 'monospace'  # each character in this font is equally wide


def make_neighbors_table_fig(similarity_matrix: np.ndarray,
                             labels: List[str],
                             test_words: List[str],
                             fontsize: int = 8,
                             ) -> plt.Figure:
    """
    Returns fig showing 10 nearest neighbors for "test_words" in "similarity_matrix"
    """

    # load data
    neighbors_mat_list = []
    col_labels_list = []
    num_cols = min(4, len(test_words))  # fixed
    num_neighbors = 10  # fixed
    for i in range(0, len(test_words), num_cols):  # split test words into even sized lists
        col_test_words = test_words[i:i + num_cols]
        neighbors_mat = np.chararray((num_neighbors, num_cols), itemsize=40, unicode=True)
        neighbors_mat[:] = ''  # initialize so that mpl can read table

        # make column
        for col_id, test_word in enumerate(col_test_words):
            tw_sims = similarity_matrix[labels.index(test_word)]

            neighbor_tuples = [(labels[w_id], token_sim) for w_id, token_sim in enumerate(tw_sims)
                               if labels[w_id] != test_word]
            neighbor_tuples = sorted(neighbor_tuples, key=lambda t: t[1], reverse=True)[:num_neighbors]

            neighbors_mat_col = [f'{t[0]:<12} sim={t[1]:.2f}' for t in neighbor_tuples
                                 if t[0] != test_word]
            neighbors_mat[:, col_id] = neighbors_mat_col

        # collect info for plotting
        neighbors_mat_list.append(neighbors_mat)
        length_diff = num_cols - len(col_test_words)
        col_labels_list.append(col_test_words + [' '] * length_diff)

    # fig
    num_tables = max(2, len(neighbors_mat_list))  # max 2 otherwise axes cannot be sliced along axis 0
    res, axes = plt.subplots(num_tables, 1,
                             figsize=(7, num_tables * (num_neighbors / 4.)),
                             dpi=config.Fig.dpi)
    for ax, neighbors_mat, col_test_words in zip_longest(axes, neighbors_mat_list, col_labels_list):
        ax.axis('off')
        if neighbors_mat is not None:  # this allows turning off of axis even when neighbors_mat list length is < 2
            table_ = ax.table(cellText=neighbors_mat, colLabels=col_test_words, loc='center')
            if fontsize is not None:
                table_.auto_set_font_size(False)
                table_.set_fontsize(fontsize)
    res.tight_layout()

    return res


NUM_WORDS = 12
NOISE = 0.3

# create random words and similarity matrix
words = [f'word-{n}' for n in range(NUM_WORDS)]
tmp1 = np.random.random((1, NUM_WORDS)).repeat(NUM_WORDS//2, axis=0) + NOISE * np.random.random((NUM_WORDS//2, NUM_WORDS))
tmp2 = np.random.random((1, NUM_WORDS)).repeat(NUM_WORDS//2, axis=0) + NOISE * np.random.random((NUM_WORDS//2, NUM_WORDS))
sim_matrix = np.vstack([tmp1, tmp2])

fig = make_neighbors_table_fig(sim_matrix, words, test_words=words[:7])
fig.show()
