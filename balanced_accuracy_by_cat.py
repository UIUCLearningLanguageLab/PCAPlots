import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src import config



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


