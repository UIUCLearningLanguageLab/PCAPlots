import numpy as np
import matplotlib.pyplot as plt

from src import config


def make_probe_probe_corr_traj_fig(probes):
    """
    Returns fig showing correlation between  probe prototype activation and activations of "alternate probes".
    """

    num_leg_per_col = 3
    probes_iter = iter(probes)
    num_probes = len(probes)
    assert num_probes % 2 != 0  # need uneven number of axes to insert legend
    num_rows = num_probes // 2 if num_probes % 2 == 0 else num_probes // 2 + 1
    num_alt_probes = len(config.Fig.ALTERNATE_PROBES)
    # load data
    probe_traj_mat_dict = {probe: np.zeros((num_alt_probes, len(model.ckpt_mb_names)))
                           for probe in probes}
    y_mins = []
    for n, mb_name in enumerate(model.ckpt_mb_names):
        model.acts_df = reload_acts_df(model.model_name, mb_name, model.hub.mode)
        probe_acts_df = model.get_multi_probe_prototype_acts_df()
        df_ids = probe_acts_df.index.isin(config.Fig.ALTERNATE_PROBES)
        alt_probes_acts = probe_acts_df.loc[df_ids].values
        for probe in probes:
            df_id = probe_acts_df.index == probe
            probe_act = probe_acts_df.iloc[df_id].values
            probe_probe_corrs = [np.corrcoef(probe_act, alt_probe_act)[1, 0]
                                 for alt_probe_act in alt_probes_acts]
            probe_traj_mat_dict[probe][:, n] = probe_probe_corrs
            if n != 0:
                y_min = min(probe_probe_corrs[1:])
                y_mins.append(y_min)
    x = model.get_data_step_axis()
    # fig
    fig, axarr = plt.subplots(num_rows, 2, figsize=(config.Fig.fig_size, 2 * num_rows), dpi=config.Fig.dpi)
    for row_id, row in enumerate(axarr):
        for ax_id, ax in enumerate(row):
            try:
                probe = next(probes_iter)
            except StopIteration:
                ax.axis('off')
                last_ax = axarr[row_id, ax_id - 1]  # make legend for last ax
                handle, label = last_ax.get_legend_handles_labels()
                ax.legend(handle, label, loc=6, ncol=num_probes // num_leg_per_col)
                continue
            else:
                ax.set_title(probe, fontsize=config.Fig.ax_label_fontsize)
                if ax_id % 2 == 0:
                    ax.set_ylabel('Correlation'.format(probe), fontsize=config.Fig.ax_label_fontsize)
                if row_id == num_rows - 1:
                    ax.set_xlabel('Mini Batch', fontsize=config.Fig.ax_label_fontsize)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_major_formatter(FuncFormatter(human_format))
                ax.set_ylim([min(y_mins), 1])
            # plot
            for alt_probe_id, alt_probe in enumerate(config.Fig.ALTERNATE_PROBES):
                traj = probe_traj_mat_dict[probe][alt_probe_id]
                ax.plot(x, traj, '-', linewidth=config.Fig.line_width, label=alt_probe)
    fig.tight_layout()
    return fig
