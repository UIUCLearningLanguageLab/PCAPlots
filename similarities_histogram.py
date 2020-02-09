import numpy as np
import matplotlib.pyplot as plt

from src import config


def make_probe_sim_hist_fig(num_bins=1000):
    """
    Returns fig showing histogram of similarities learned by "model"
    """

    # load data
    probes_acts_df = model.get_multi_probe_prototype_acts_df()
    probe_simmat = cosine_similarity(probes_acts_df.values)
    probe_simmat[np.tril_indices(probe_simmat.shape[0], -1)] = np.nan
    probe_simmat_values = probe_simmat[~np.isnan(probe_simmat)]

    # fig
    fig, ax = plt.subplots(figsize=(config.Fig.fig_size, 3), dpi=config.Fig.dpi)
    ax.set_xlabel('Similarity', fontsize=config.Fig.ax_label_fontsize)
    ax.set_ylabel('Frequency', fontsize=config.Fig.ax_label_fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.grid(True)

    # plot
    step_size = 1.0 / num_bins
    bins = np.arange(-1, 1, step_size)
    hist, _ = np.histogram(probe_simmat_values, bins=bins)
    x_binned = bins[:-1]
    ax.plot(x_binned, hist, '-', linewidth=config.Fig.line_width, c='black')
    plt.tight_layout()

    return fig

