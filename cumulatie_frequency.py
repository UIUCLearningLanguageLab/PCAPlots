from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from collections import Counter
import random

from src import config
from src.utils import human_format


def make_cumulative_frequency_fig(words: List[str],
                                  corpus_partitions: List[List[str]],
                                  ) -> plt.Figure:
    """
    Returns fig showing time course of cumulative frequency of "words" across "corpus_partitions"
    """

    palette = iter(sns.color_palette("hls", len(words)))

    # count
    num_parts = len(corpus_partitions)
    part2w2f = {n: Counter(corpus_partitions[n]) for n in range(num_parts)}

    # collect cum. frequencies for each word
    x = np.arange(num_parts)
    xys = []
    for w in words:

        frequencies = [part2w2f[n][w] for n in range(num_parts)]
        y = np.cumsum(frequencies)

        print(w)
        print(y)

        # get last frequency for figure annotation
        last_y, last_x = y[-1], x[-1]

        xys.append((x, y, last_x, last_y, w))

    # fig
    res, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.set_xlabel('Corpus Location', fontsize=config.Fig.ax_label_fontsize)
    ax.set_ylabel('Cumulative Frequency', fontsize=config.Fig.ax_label_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))

    # plot
    y_thr = np.max([xy[3] for xy in xys]) / 10  # threshold is at third from max
    for (x, y, last_x, last_y, w) in xys:
        ax.plot(x, y, '-', linewidth=1.0, c=next(palette))
        if last_y > y_thr:
            plt.annotate(w, xy=(last_x, last_y),
                         xytext=(0, 0), textcoords='offset points',
                         va='center', fontsize=config.Fig.legend_fontsize, bbox=dict(boxstyle='round', fc='w'))

    return res


NUM_PARTS = 256
PART_SIZE = 100
NUM_WORDS = 50

words = [f'w{i}' for i in range(NUM_WORDS)]
parts = [random.choices(random.choices(words, k=10), k=PART_SIZE)
         for _ in range(NUM_PARTS)]
fig = make_cumulative_frequency_fig(words[:5], parts)
fig.show()