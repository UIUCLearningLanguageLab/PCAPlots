import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src import config


def make_principal_comps_table_fig(pc_id):
    """
    Returns fig showing table of words loading highest and lowest on a principal component
    """

    num_cols = 2
    col_labels = ['Low End', 'High End']

    # load data
    acts_mat = model.term_acts_mat if config.Fig.SVD_IS_TERMS else model.get_multi_probe_prototype_acts_df().values
    tokens = model.hub.train_terms.types if config.Fig.SVD_IS_TERMS else model.hub.probe_store.types
    pca_model = PCA(n_components=config.Fig.NUM_PCS)
    pcs = pca_model.fit_transform(acts_mat)
    expl_var_perc = np.asarray(pca_model.explained_variance_ratio_) * 100
    pc = pcs[:, pc_id]

    # sort and filter
    sorted_pc, sorted_tokens = list(zip(*sorted(zip(pc, tokens), key=lambda i: i[0])))
    col0_strs = ['{} {:.2f} (freq: {:,})'.format(term, loading, sum(model.hub.term_part_freq_dict[term]))
                 for term, loading in zip(
            sorted_tokens[:config.Fig.NUM_PCA_LOADINGS], sorted_pc[:config.Fig.NUM_PCA_LOADINGS])
                 if sum(model.hub.term_part_freq_dict[term]) > config.Fig.PCA_FREQ_THR]
    col1_strs = ['{} {:.2f} ({:,})'.format(term, loading, sum(model.hub.term_part_freq_dict[term]))
                 for term, loading in zip(
            sorted_tokens[-config.Fig.NUM_PCA_LOADINGS:][::-1], sorted_pc[-config.Fig.NUM_PCA_LOADINGS:][::-1])
                 if sum(model.hub.term_part_freq_dict[term]) > config.Fig.PCA_FREQ_THR]

    # make probes_mat
    max_rows = max(len(col0_strs), len(col1_strs))
    probes_mat = np.chararray((max_rows, num_cols), itemsize=40, unicode=True)
    probes_mat[:] = ''  # initialize so that mpl can read table
    probes_mat[:len(col0_strs), 0] = col0_strs
    probes_mat[:len(col1_strs), 1] = col1_strs

    # fig
    fig, ax = plt.subplots(figsize=(config.Fig.fig_size, 0.25 * max_rows), dpi=config.Fig.dpi)
    ax.set_title('Principal Component {} ({:.2f}% var)'.format(
        pc_id, expl_var_perc[pc_id]), fontsize=config.Fig.ax_label_fontsize)
    ax.axis('off')

    # plot
    table_ = ax.table(cellText=probes_mat, colLabels=col_labels,
                      loc='center', colWidths=[0.3] * num_cols)
    table_.auto_set_font_size(False)
    table_.set_fontsize(8)
    fig.tight_layout()

    return fig


def make_principal_comps_line_fig(pc_id):
    """
    Returns fig showing loadings on a specified principal component
    """

    # load data
    acts_mat = model.term_acts_mat if config.Fig.SVD_IS_TERMS else model.get_multi_probe_prototype_acts_df().values
    pca_model = PCA(n_components=config.Fig.NUM_PCS)
    pcs = pca_model.fit_transform(acts_mat)
    pc = pcs[:, pc_id]

    # fig
    fig, ax = plt.subplots(figsize=(config.Fig.fig_size, 3), dpi=config.Fig.dpi)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')

    # plot
    ax.plot(sorted(pc), '-', linewidth=config.Fig.LINEWIDTH, color='black')
    ax.set_xticklabels([])
    ax.set_xlabel('Token IDs', fontsize=config.Fig.ax_label_fontsize)
    ax.set_ylabel('PC {} Loading'.format(pc_id), fontsize=config.Fig.ax_label_fontsize)
    fig.tight_layout()

    return fig
