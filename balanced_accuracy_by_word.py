from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import random

from src import config


def make_ba_breakdown_annotated_fig(word2ba: Dict[str, float],
                                    category2words: Dict[str, List[str]]
                                    ) -> plt.Figure:
    """
    Returns fig showing each word's balanced accuracy in relation to all others
    """

    cats_per_axis = 16
    x_cycle = cycle(range(cats_per_axis))

    avg_ba = np.mean(list(word2ba.values()))

    # calculate position of each word in figure
    cats_sorted_by_ba = sorted(category2words.keys(),
                               key=lambda c: np.mean([word2ba[w] for w in category2words[c]]))
    xys = []
    for cat in cats_sorted_by_ba:
        words_in_category = category2words[cat]
        x = [next(x_cycle)] * len(words_in_category)
        y = [word2ba[w] for w in words_in_category]
        xys.append((x, y, words_in_category))

    # fig
    num_axes = len(category2words) // cats_per_axis + 1
    res, axes = plt.subplots(num_axes, figsize=(12, 4 * num_axes), dpi=config.Fig.dpi)
    for n, ax in enumerate(axes):

        # truncate data
        xys_truncated = xys[n * cats_per_axis: (n + 1) * cats_per_axis]
        cats_sorted_by_ba_truncated = cats_sorted_by_ba[n * cats_per_axis: (n + 1) * cats_per_axis]

        # axis
        ax.set_ylabel('Balanced Accuracy',  fontsize=config.Fig.ax_label_fontsize)
        ax.set_xticks(np.arange(cats_per_axis), minor=False)  # shifts x tick labels right
        ax.set_xticklabels(cats_sorted_by_ba_truncated, minor=False, fontsize=config.Fig.tick_label_fontsize,
                           rotation=90)
        ax.set_xlim([-1, cats_per_axis])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=avg_ba, alpha=config.Fig.FILL_ALPHA, c='grey', linestyle='--', zorder=1)

        # plot
        annotated_y_ints_long_words_prev_cat = []
        for (x, y, words_in_cat) in xys_truncated:
            ax.plot(x, y, 'b.', alpha=1.0)  # this needs to be plot for annotation to work

            # annotate points
            annotated_y_ints = []
            annotated_y_ints_long_words_curr_cat = []
            for x_, y_, word in zip(x, y, words_in_cat):
                y_int = round(y_, 2)

                # if annotation coordinate exists or is affected by long word from previous cat, skip to next word
                if y_int not in annotated_y_ints and y_int not in annotated_y_ints_long_words_prev_cat:
                    ax.annotate(word, xy=(x_, y_int), xytext=(2, 0), textcoords='offset points', va='bottom',
                                fontsize=7)
                    annotated_y_ints.append(y_int)
                    if len(word) > 7:
                        annotated_y_ints_long_words_curr_cat.append(y_int)
            annotated_y_ints_long_words_prev_cat = annotated_y_ints_long_words_curr_cat

    return res


NUM_WORDS = 600
NUM_CATEGORIES = 30

# create random words and categories
words = [f'word-{n}' for n in range(NUM_WORDS)]
word2ba = {w: random.random() for w in words}
cat_size = NUM_WORDS // NUM_CATEGORIES
cat2words = {f'cat-{n}': words[cat_size * n: cat_size * n + cat_size] for n in range(NUM_CATEGORIES)}

fig = make_ba_breakdown_annotated_fig(word2ba, cat2words)
fig.show()
