import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src import config


def make_ba_breakdown_annotated_fig(context_type):
    """
    Returns fig showing ranking of each probe's avg_probe_fs broken down by category
    """
    cats_per_axis = 16
    x_cycler = cycle(range(cats_per_axis))

    # load data
    cats_sorted_by_fs = model.sort_cats_by_ba(context_type)
    sorted_cat_avg_probe_fs_lists = model.get_sorted_cat_avg_probe_ba_lists(cats_sorted_by_fs, 'ordered')
    mean_fs = np.mean(model.avg_probe_fs_o_list)
    xys = []
    for cat, cat_avg_probe_fs_list in zip(cats_sorted_by_fs, sorted_cat_avg_probe_fs_lists):
        cat_probes = model.hub.probe_store.cat_probe_list_dict[cat]
        x = [next(x_cycler)] * len(cat_probes)
        y = cat_avg_probe_fs_list
        xys.append((x, y, cat_probes))

    # fig
    num_axes = len(model.hub.probe_store.cats) // cats_per_axis + 1
    fig, axarr = plt.subplots(num_axes, figsize=(config.Fig.fig_size, 6 * num_axes), dpi=config.Fig.dpi)
    for n, ax in enumerate(axarr):

        # truncate data
        xys_truncated = xys[n * cats_per_axis: (n + 1) * cats_per_axis]
        cats_sorted_by_fs_truncated = cats_sorted_by_fs[n * cats_per_axis: (n + 1) * cats_per_axis]

        # axis
        ax.set_ylabel('Avg Probe Balanced Accuracy ({})'.format(context_type),
                      fontsize=config.Fig.ax_label_fontsize)
        ax.set_xticks(np.arange(cats_per_axis), minor=False)  # shifts xtick labels right
        ax.set_xticklabels(cats_sorted_by_fs_truncated, minor=False, fontsize=config.Fig.TICKLABEL_FONT_SIZE,
                           rotation=90)
        ax.set_xlim([0, cats_per_axis])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=mean_fs, alpha=config.Fig.FILL_ALPHA, c='grey', linestyle='--', zorder=1)

        # plot
        annotated_y_ints_long_words_prev_cat = []
        for (x, y, cat_probes) in xys_truncated:
            ax.plot(x, y, 'b.', alpha=0)  # this needs to be plot for annotation to work

            # annotate points
            annotated_y_ints = []
            annotated_y_ints_long_words_curr_cat = []
            for x_, y_, probe in zip(x, y, cat_probes):
                y_int = int(y_)

                # if annotation coordinate exists or is affected by long word from previous cat, skip to next probe
                if y_int not in annotated_y_ints and y_int not in annotated_y_ints_long_words_prev_cat:
                    ax.annotate(probe, xy=(x_, y_int), xytext=(2, 0), textcoords='offset points', va='bottom',
                                fontsize=7)
                    annotated_y_ints.append(y_int)
                    if len(probe) > 7:
                        annotated_y_ints_long_words_curr_cat.append(y_int)
            annotated_y_ints_long_words_prev_cat = annotated_y_ints_long_words_curr_cat
    plt.tight_layout()

    return fig


def sort_cats_by_group1_ba(context_type):
    cats = model_groups[0][0].hub.probe_store.cats
    num_cats = len(cats)
    mat = np.zeros((len(model_groups[0]), num_cats))
    for n, model in enumerate(model_groups[0]):
        mat[n, :] = [np.mean(cat_avg_probe_ba_list) for cat_avg_probe_ba_list
                     in model.get_sorted_cat_avg_probe_ba_lists(model.hub.probe_store.cats, context_type)]
    avgs = mat.mean(axis=0).tolist()
    _, tup = zip(*sorted(zip(avgs, cats), key=lambda t: t[0]))
    result = list(tup)
    return result


def make_cat_ba_mat_fig(context_type, sg_embed_size=512):
    """
    Returns fig showing heatmap of F1-scores from multiple models broken down by category
    """
    num_models = len([model for models in model_groups for model in models])
    cats = model_groups[0][0].hub.probe_store.cats
    hub_mode = model_groups[0][0].hub.mode
    num_cats = len(cats)
    # load data
    sorted_cats = sort_cats_by_group1_ba(context_type)
    group_names = []
    cat_ba_mat = np.zeros((num_models, num_cats))
    row_ids = iter(range(num_models))
    for model_desc, models in zip(model_descs, model_groups):
        for model_id, model in enumerate(models):
            group_name = model_desc.replace('\n', ' ').split('=')[-1]
            group_names.append(group_name)
            sorted_cat_avg_probe_ba_lists = model.get_sorted_cat_avg_probe_ba_lists(sorted_cats, context_type)
            cat_ba_mat[next(row_ids), :] = [np.mean(cat_avg_probe_ba_list)
                                            for cat_avg_probe_ba_list in sorted_cat_avg_probe_ba_lists]
    # load sg data
    path = config.Fig.SG_DIR / 'sg_df_{}_{}.csv'.format(hub_mode, sg_embed_size)
    if path.exists():
        df_sg = pd.read_csv(path)
        sg_cat_ba_mat = df_sg.groupby('cat').mean().transpose()[sorted_cats].values
        num_sgs = len(sg_cat_ba_mat)
        group_names += ['skip-gram'] * num_sgs
        cat_ba_mat = np.vstack((cat_ba_mat, sg_cat_ba_mat))
    else:
        num_sgs = 0

    # fig
    fig, ax = plt.subplots(figsize=(config.Fig.fig_size, 6))
    plt.title('context_type="{}"'.format(context_type), fontsize=config.Fig.ax_label_fontsize)

    # plot
    sns.heatmap(cat_ba_mat, ax=ax, square=True, annot=False,
                annot_kws={"size": 6}, cbar_kws={"shrink": .5},
                vmin=0, vmax=1, cmap='jet', fmt='d')

    # colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['0', '0.5', '1'])
    cbar.set_label('Avg Category Probe Balanced Accuracy')

    # ax (needs to be below plot for axes to be labeled)
    ax.set_yticks(np.arange(num_models + num_sgs) + 0.5)
    ax.set_xticks(np.arange(num_cats) + 0.5)
    ax.set_yticklabels(group_names, rotation=0)
    ax.set_xticklabels(sorted_cats, rotation=90)
    for t in ax.texts:
        t.set_text(t.get_text() + "%")
    plt.tight_layout()

    return fig


def make_ba_by_cat_fig(context_type, sg_embed_size=512):
    """
    Returns fig showing model group averages of probes balanced accuracy broken down by category
    """
    num_model_groups = len(model_groups)
    palette = cycle(sns.color_palette("hls", num_model_groups))
    cats = model_groups[0][-1].hub.probe_store.cats
    hub_mode = model_groups[0][-1].hub.mode
    num_cats = len(cats)
    # load data
    sorted_cats = sort_cats_by_group1_ba(context_type)
    xys = []
    for models, model_desc in zip(model_groups, model_descs):
        cat_ba_mat = np.zeros((len(models), num_cats))
        for model_id, model in enumerate(models):
            sorted_cat_avg_probe_ba_lists = model.get_sorted_cat_avg_probe_ba_lists(sorted_cats, context_type)
            cat_ba_mat[model_id, :] = [np.mean(cat_avg_probe_ba_list)
                                       for cat_avg_probe_ba_list in sorted_cat_avg_probe_ba_lists]
        x = range(num_cats)
        y = np.mean(cat_ba_mat, axis=0)
        sem = stats.sem(cat_ba_mat, axis=0)
        num_models = len(models)
        xys.append((x, y, sem, model_desc, num_models))

    # fig
    fig, ax = plt.subplots(figsize=(config.Fig.fig_size, 6))
    plt.title('context_type="{}"'.format(context_type), fontsize=config.Fig.ax_label_fontsize)
    ax.set_ylabel('Balanced Accuracy (+/-SEM)', fontsize=config.Fig.ax_label_fontsize, labelpad=0.0)
    ax.set_xticks(np.arange(num_cats), minor=False)
    ax.set_xticklabels(sorted_cats, minor=False, fontsize=config.Fig.TICKLABEL_FONT_SIZE, rotation=90)
    ax.set_xlim([0, len(cats)])
    ax.set_ylim([0.5, 1.0])
    ax.set_axisbelow(True)  # put grid under plot lines
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plot
    for (x, y, sem, model_desc, num_models) in xys:
        color = next(palette)
        ax.plot(x, y, '-', color=color, linewidth=config.Fig.LINEWIDTH,
                label='{} n={}'.format(model_desc, num_models))
        ax.fill_between(x, np.add(y, sem), np.subtract(y, sem), alpha=config.Fig.FILL_ALPHA, color='grey')

    # plot sg
    path = config.Fig.SG_DIR / 'sg_df_{}_{}.csv'.format(hub_mode, sg_embed_size)
    if path.exists():
        df_sg = pd.read_csv(path)
        df_sg['avg_probe_ba_mean'] = df_sg.filter(regex="avg_probe_ba_\d").mean(axis=1)
        df_sg['avg_probe_ba_sem'] = df_sg.filter(regex="avg_probe_ba_\d").sem(axis=1)
        num_sgs = len(df_sg.filter(regex="avg_probe_ba_\d").columns)
        cat_y_dict = df_sg.groupby('cat').mean().to_dict()['avg_probe_ba_mean']
        cat_sem_dict = df_sg.groupby('cat').mean().to_dict()['avg_probe_ba_sem']
        y = [cat_y_dict[cat] for cat in sorted_cats]
        sem = [cat_sem_dict[cat] for cat in sorted_cats]
        x = range(num_cats)
        ax.plot(x, y, '-', color='black', linewidth=config.Fig.LINEWIDTH,
                label='skipgram num_h{} n={}'.format(sg_embed_size, num_sgs))
        ax.fill_between(x, np.add(y, sem), np.subtract(y, sem), alpha=config.Fig.FILL_ALPHA, color='grey')
        sg_avg_probe_bas = df_sg.filter(regex="avg_probe_ba_\d").mean(axis=0)
        print('Skip-gram avg_probe_ba mean across models:', sg_avg_probe_bas.mean())
        print('Skip-gram avg_probe_ba sem across models:', sg_avg_probe_bas.sem())
    plt.tight_layout()
    add_single_legend(ax, model_descs, y_offset=-0.60)  # TODO do this in other places too

    return fig