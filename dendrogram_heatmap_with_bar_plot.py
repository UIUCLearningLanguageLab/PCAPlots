from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src import config


def make_probe_sim_dh_fig_jw(probes,
                             num_colors=None,
                             label_doc_id=True,
                             vmin=0.90, vmax=1.0,
                             cluster=False):
    """
    Returns fig showing dendrogram heatmap of similarity matrix of "probes"
    """

    num_probes = len(probes)
    assert num_probes > 1

    # load data
    probes_acts_df = model.get_multi_probe_prototype_acts_df()
    probe_ids = [model.hub.probe_store.probe_id_dict[probe] for probe in probes]
    probes_acts_df_filtered = probes_acts_df.iloc[probe_ids]
    probe_simmat = cosine_similarity(probes_acts_df_filtered.values)
    print('Probe simmat  min: {} max {}'.format(np.min(probe_simmat), np.max(probe_simmat)))
    print('Fig  min: {} max {}'.format(vmin, vmax))

    # fig
    fig, ax_heatmap = plt.subplots(figsize=(11, 7), dpi=config.Fig.dpi)
    if label_doc_id:
        plt.title('Trained on {:,} terms'.format(model.mb_size * int(model.mb_name)),
                  fontsize=config.Fig.ax_label_fontsize)
    ax_heatmap.yaxis.tick_right()
    divider = make_axes_locatable(ax_heatmap)
    ax_dend = divider.append_axes("bottom", 0.8, pad=0.0)  # , sharex=ax_heatmap)
    ax_colorbar = divider.append_axes("left", 0.1, pad=0.4)
    ax_freqs = divider.append_axes("right", 4.0, pad=0.0)  # , sharey=ax_heatmap)

    # dendrogram
    ax_dend.set_frame_on(False)
    lnk0 = linkage(pdist(probe_simmat))
    if num_colors is None or num_colors <= 1:
        left_threshold = -1
    else:
        left_threshold = 0.5 * (lnk0[1 - num_colors, 2] +
                                lnk0[-num_colors, 2])
    dg0 = dendrogram(lnk0, ax=ax_dend,
                     orientation='bottom',
                     color_threshold=left_threshold,
                     no_labels=True,
                     no_plot=not cluster)
    if cluster:
        # Reorder the values in x to match the order of the leaves of the dendrograms
        z = probe_simmat[dg0['leaves'], :]  # sorting rows
        z = z[:, dg0['leaves']]  # sorting columns for symmetry
        simmat_labels = np.array(probes)[dg0['leaves']]
    else:
        z = probe_simmat
        simmat_labels = probes

    # probe freq bar plot
    doc_id = model.doc_id if label_doc_id else model.num_docs  # faster if not label_doc_id
    probe_freqs = [sum(model.hub.term_part_freq_dict[probe][:doc_id]) * model.num_iterations
                   for probe in simmat_labels]
    y = range(num_probes)
    ax_freqs.barh(y, probe_freqs, color='black')
    ax_freqs.set_xlabel('Freq')
    ax_freqs.set_xlim([0, config.Fig.PROBE_FREQ_YLIM])
    ax_freqs.set_xticks([0, config.Fig.PROBE_FREQ_YLIM])
    ax_freqs.set_xticklabels([0, config.Fig.PROBE_FREQ_YLIM])
    ax_freq_lim0 = ax_freqs.get_ylim()[0] + 0.5
    ax_freq_lim1 = ax_freqs.get_ylim()[1] - 0.5
    ax_freqs.set_ylim([ax_freq_lim0, ax_freq_lim1])  # shift ticks to match heatmap
    ax_freqs.yaxis.set_ticks(y)
    ax_freqs.yaxis.set_ticklabels(simmat_labels, color='white')

    # heatmap
    max_extent = ax_dend.get_xlim()[1]
    im = ax_heatmap.imshow(z[::-1], aspect='auto',
                           cmap=plt.cm.jet,
                           extent=(0, max_extent, 0, max_extent),
                           vmin=vmin, vmax=vmax)

    # colorbar
    cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='vertical')
    cb.ax.set_xticklabels([vmin, vmax], fontsize=config.Fig.ax_label_fontsize, rotation=90)
    cb.set_label('Correlation Coefficient', labelpad=-10, fontsize=config.Fig.ax_label_fontsize, rotation=90)

    # set heatmap ticklabels
    ax_heatmap.xaxis.set_ticks([])
    ax_heatmap.xaxis.set_ticklabels([])
    ax_heatmap.yaxis.set_ticks([])
    ax_heatmap.yaxis.set_ticklabels([])

    # Hide all tick lines
    lines = (ax_heatmap.xaxis.get_ticklines() +
             ax_heatmap.yaxis.get_ticklines() +
             ax_dend.xaxis.get_ticklines() +
             ax_dend.yaxis.get_ticklines())
    plt.setp(lines, visible=False)

    # set label rotation and fontsize
    x_labels = ax_heatmap.xaxis.get_ticklabels()
    plt.setp(x_labels, rotation=-90)
    plt.setp(x_labels, fontsize=config.Fig.ax_label_fontsize)
    y_labels = ax_heatmap.yaxis.get_ticklabels()
    plt.setp(y_labels, rotation=0)
    plt.setp(y_labels, fontsize=config.Fig.ax_label_fontsize)

    # make dendrogram labels invisible
    plt.setp(ax_dend.get_yticklabels() + ax_dend.get_xticklabels(),
             visible=False)
    fig.tight_layout()

    return fig