    def make_cat_cluster_fig(cat, bottom_off=False, max_probes=20, metric='cosine', x_max=config.Fig.CAT_CLUSTER_XLIM):
        """
        Returns fig showing hierarchical clustering of probes in single category
        """
        start = time.time()
        # load data
        cat_prototypes_df = model.get_single_cat_probe_prototype_acts_df(cat)
        if len(cat_prototypes_df) > max_probes:
            ids = np.random.choice(len(cat_prototypes_df) - 1, max_probes, replace=False)
            cat_prototypes_df = cat_prototypes_df.iloc[ids]
            probes_in_cat = cat_prototypes_df.index
        else:
            probes_in_cat = cat_prototypes_df.index.tolist()
        # fig
        rcParams['lines.linewidth'] = 2.0
        fig, ax = plt.subplots(figsize=(config.Fig.MAX_FIG_WIDTH, 4), dpi=config.Fig.DPI)
        # dendrogram
        dist_matrix = pdist(cat_prototypes_df.values, metric=metric)
        linkages = linkage(dist_matrix, method='complete')
        dendrogram(linkages,
                   ax=ax,
                   leaf_label_func=lambda x: probes_in_cat[x],
                   orientation='right',
                   leaf_font_size=8)
        ax.set_title(cat)
        ax.set_xlim([0, x_max])
        ax.tick_params(axis='both', which='both', top='off', right='off', left='off')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if bottom_off:
            ax.xaxis.set_ticklabels([])  # hides ticklabels
            ax.tick_params(axis='both', which='both', bottom='off')
            ax.spines['bottom'].set_visible(False)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    figs = [make_cat_cluster_fig(cat) for cat in model.hub.probe_store.cats]
    return figs


def make_multi_cat_clust_fig(cats, metric='cosine'):  # TODO make into config
    """
    Returns fig showing hierarchical clustering of probes from multiple categories
    """
    start = time.time()
    # load data
    df = pd.DataFrame(pd.concat((model.get_single_cat_probe_prototype_acts_df(cat) for cat in cats), axis=0))
    cat_acts_mat = df.values
    cats_probe_list = df.index
    # fig
    rcParams['lines.linewidth'] = 2.0
    fig, ax = plt.subplots(figsize=(config.Fig.MAX_FIG_WIDTH, 5 * len(cats)), dpi=config.Fig.DPI)
    # dendrogram
    dist_matrix = pdist(cat_acts_mat, metric)
    linkages = linkage(dist_matrix, method='complete')
    dendrogram(linkages,
               ax=ax,
               labels=cats_probe_list,
               orientation='right',
               leaf_font_size=10)
    ax.tick_params(axis='both', which='both', top='off', right='off', left='off')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
    return fig
