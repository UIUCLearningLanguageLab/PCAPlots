import numpy as np
import matplotlib.pyplot as plt

from src import config


def make_probes_by_act_fig(hidden_unit_ids, num_probes=20):
    """
    Returns fig showing probes with highest activation at "hidden_unit_ids"
    """
    hidden_unit_ids = [int(i) for i in hidden_unit_ids]

    # load data
    multi_acts_df = model.get_multi_probe_prototype_acts_df()
    acts_mat = multi_acts_df.values

    # make probes_by_acts_mat
    probe_by_acts_mat_list = []
    col_labels_list = []
    num_cols = 5  # fixed
    for i in range(0, len(hidden_unit_ids), num_cols):  # split into even sized lists
        hidden_unit_ids_ = hidden_unit_ids[i:i + num_cols]
        probes_by_acts_mat = np.chararray((num_probes, num_cols), itemsize=20, unicode=True)
        probes_by_acts_mat[:] = ''  # initialize so that mpl can read table
        # make column
        for col_id, hidden_unit_id in enumerate(hidden_unit_ids_):
            acts_mat_col = acts_mat[:, hidden_unit_id]
            tups = [(model.hub.probe_store.types[probe_id], act) for probe_id, act in enumerate(acts_mat_col)]
            sorted_tups = sorted(tups, key=lambda i: i[1], reverse=True)[:num_probes]
            probe_by_act_mat_col = ['{:>15} {:.2f}'.format(tup[0], tup[1]) for tup in sorted_tups]
            probes_by_acts_mat[:, col_id] = probe_by_act_mat_col
        # collect info for plotting
        probe_by_acts_mat_list.append(probes_by_acts_mat)
        length_diff = num_cols - len(hidden_unit_ids_)
        for i in range(length_diff):
            hidden_unit_ids_.append(' ')  # add space so table can be read properly
        col_labels_list.append(['hidden unit #{}'.format(hidden_unit_id)
                                if hidden_unit_id != ' ' else ' '
                                for hidden_unit_id in hidden_unit_ids_])

    # fig
    num_tables = len(probe_by_acts_mat_list)
    if num_tables == 1:
        fig, axarr = plt.subplots(1, 1,
                                  figsize=(config.Fig.fig_size, num_tables * (num_probes / 4.)),
                                  dpi=config.Fig.dpi)
        axarr = [axarr]
    else:
        fig, axarr = plt.subplots(num_tables, 1,
                                  figsize=(config.Fig.fig_size, num_tables * (num_probes / 4.)),
                                  dpi=config.Fig.dpi)
    for ax, probes_by_acts_mat, col_labels in zip(axarr, probe_by_acts_mat_list, col_labels_list):
        ax.axis('off')
        table_ = ax.table(cellText=probes_by_acts_mat, colLabels=col_labels, loc='center')
        table_.auto_set_font_size(False)
        table_.set_fontsize(8)
    fig.tight_layout()

    return fig
