    def make_cat_cluster_fig(cat, bottom_off=False, max_probes=20, metric='cosine', x_max=FigsConfigs.CAT_CLUSTER_XLIM):
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
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 4), dpi=FigsConfigs.DPI)
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
