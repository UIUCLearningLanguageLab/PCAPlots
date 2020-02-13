from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.utils import human_format


def make_cosine_timeline_fig(test_embeddings: np.ndarray,
                             reference_embeddings: np.ndarray,
                             test_labels: List[str],
                             reference_labels: List[str],
                             ) -> plt.Figure:
    """
    Returns fig showing time course of correlation between embeddings and reference embeddings
    """

    assert np.ndim(test_embeddings) == 3  # (ticks, words, embed size)
    assert np.ndim(reference_embeddings) == 3

    # calculate correlations between each reference and test embedding, for each time step
    # pairwise cosine return shape (num test, num reference) - correlations is (num ticks, num test, num ref)
    correlations = np.asarray([cosine_similarity(test_embeddings[tick], reference_embeddings[tick])
                               for tick in range(len(reference_embeddings))])
    print(correlations)
    print(correlations.shape)

    # fig
    num_test = len(test_labels)
    assert num_test % 2 != 0  # need odd number of axes to insert legend into an empty axis
    num_rows = num_test // 2 if num_test % 2 == 0 else num_test // 2 + 1
    test_label_iterator = iter(test_labels)
    res, axes = plt.subplots(num_rows, 2, figsize=(7, 2 * num_rows), dpi=config.Fig.dpi)
    for axes_row_id, axes_row in enumerate(axes):
        for axes_col_id, ax in enumerate(axes_row):

            # a single axis belongs to a single test word

            # axis
            try:
                test_label = next(test_label_iterator)
            except StopIteration:
                ax.axis('off')
                last_ax = axes[axes_row_id, axes_col_id - 1]  # make legend for last ax
                handle, label = last_ax.get_legend_handles_labels()
                ax.legend(handle, label, loc=6, ncol=num_test // 3, frameon=False)
                continue
            else:
                ax.set_title(test_label, fontsize=config.Fig.ax_label_fontsize)
                if axes_col_id % 2 == 0:
                    ax.set_ylabel('Cosine'.format(test_label), fontsize=config.Fig.ax_label_fontsize)
                if axes_row_id == num_rows - 1:
                    ax.set_xlabel('Training Time', fontsize=config.Fig.ax_label_fontsize)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_major_formatter(FuncFormatter(human_format))
                ax.set_ylim([-1.0, 1.0])

            # plot
            for ref_label in reference_labels:
                # need shape (num ticks, num ref)
                y = correlations[:, test_labels.index(test_label), reference_labels.index(ref_label)]
                ax.plot(y, '-', linewidth=config.Fig.line_width, label=ref_label)

    return res


NUM_TICKS = 12
NUM_TEST_WORDS = 5
NUM_REFERENCE_WORDS = 2
EMBED_SIZE = 8

# create random words and random embeddings
test_words = [f'word-{n}' for n in range(NUM_TEST_WORDS)]
reference_words = [f'word-{n}' for n in range(NUM_TEST_WORDS, NUM_TEST_WORDS + NUM_REFERENCE_WORDS)]
test_embeddings = np.stack([np.random.random((NUM_TEST_WORDS, EMBED_SIZE)) for _ in range(NUM_TICKS)])
reference_embeddings = np.stack([np.random.random((NUM_REFERENCE_WORDS, EMBED_SIZE)) for _ in range(NUM_TICKS)])

print(test_embeddings.shape)
print(reference_embeddings.shape)

fig = make_cosine_timeline_fig(test_embeddings, reference_embeddings, test_words, reference_words)
fig.show()
