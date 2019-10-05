    def make_principal_comps_termgroup_heatmap_fig(tokengroup_tokens_dict,
                                                   label_expl_var=False):
        """
        Returns fig showing heatmap of loadings of probes on principal components by "tokengroup"
        """
        start = time.time()

        tokengroup_list = sorted(tokengroup_tokens_dict.keys())
        num_tokengroups = len(tokengroup_list)
        pc_labels = model.hub.train_terms.types
        # load data
        acts_mat = model.term_acts_mat
        pca_model = sklearnPCA(n_components=FigsConfigs.NUM_PCS)
        pcs = pca_model.fit_transform(acts_mat)
        expl_vars = np.asarray(pca_model.explained_variance_ratio_) * 100
        # make pca_mat
        pca_mat = np.zeros((num_tokengroups, FigsConfigs.NUM_PCS))
        for pc_id, pc in enumerate(pcs.transpose()):
            for tokengroup_id, tokengroup in enumerate(tokengroup_list):
                tokens = tokengroup_tokens_dict[tokengroup]
                loadings = [loading for loading, token in zip(pc, pc_labels) if token in tokens]
                pca_mat[tokengroup_id, pc_id] = sum(loadings) / len(tokens)
        # fig
        fig, ax_heatmap = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 0.2 * num_tokengroups),
                                       dpi=FigsConfigs.DPI)
        divider = make_axes_locatable(ax_heatmap)
        ax_colorbar = divider.append_axes("right", 0.2, pad=0.1)
        # cluster rows
        lnk0 = linkage(pdist(pca_mat))
        dg0 = dendrogram(lnk0,
                         no_plot=True)
        z = pca_mat[dg0['leaves'], :]
        # heatmap
        max_extent = ax_heatmap.get_ylim()[1]
        vmin, vmax = round(np.min(z), 1), round(np.max(z), 1)
        im = ax_heatmap.imshow(z[::-1], aspect='auto',
                               cmap=plt.cm.jet,
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='vertical')
        cb.ax.set_xticklabels([vmin, vmax], fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        cb.set_label('Average Loading', labelpad=-10, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # set heatmap ticklabels
        xlim = ax_heatmap.get_xlim()[1]
        halfxw = 0.5 * xlim / FigsConfigs.NUM_PCS
        ax_heatmap.xaxis.set_ticks(np.linspace(halfxw, xlim - halfxw, FigsConfigs.NUM_PCS))
        if label_expl_var:
            ax_heatmap.xaxis.set_ticklabels(['PC {} ({:.1f} %)'.format(pc_id + 1, expl_var)
                                             for pc_id, expl_var in zip(range(FigsConfigs.NUM_PCS), expl_vars)])
        else:
            ax_heatmap.xaxis.set_ticklabels(['PC {}'.format(pc_id + 1)
                                             for pc_id in range(FigsConfigs.NUM_PCS)])
        ylim = ax_heatmap.get_ylim()[1]
        halfyw = 0.5 * ylim / num_tokengroups
        ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, num_tokengroups))
        ax_heatmap.yaxis.set_ticklabels(np.array(tokengroup_list)[dg0['leaves']])
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # set label rotation and fontsize
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        fig.subplots_adjust(bottom=0.2)  # make room for tick labels
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_principal_comps_item_heatmap_fig(pca_item_list, label_expl_var=True):
        """
        Returns fig showing heatmap of loadings of probes on principal components by custom list of words
        """
        start = time.time()

        # load data
        num_items = len(pca_item_list)
        acts_mat = model.term_acts_mat
        pca_model = sklearnPCA(n_components=FigsConfigs.NUM_PCS)
        pcs = pca_model.fit_transform(acts_mat)
        expl_vars = np.asarray(pca_model.explained_variance_ratio_) * 100
        # make item_pca_mat
        item_pca_mat = np.zeros((len(pca_item_list), FigsConfigs.NUM_PCS))
        for pc_id, pc in enumerate(pcs.transpose()):
            for item_id, item in enumerate(pca_item_list):
                try:
                    item_loading = [loading for loading, token in zip(pc, model.hub.train_terms.types)
                                    if token == item][0]
                except IndexError:  # if item not in vocab
                    item_loading = 0
                    print('Not in vocabulary: "{}"'.format(item))
                item_pca_mat[item_id, pc_id] = item_loading
        # fig
        width = min(FigsConfigs.MAX_FIG_WIDTH, FigsConfigs.NUM_PCS * 1.5 - 1)
        fig, ax_heatmap = plt.subplots(figsize=(width, 0.2 * num_items + 2), dpi=FigsConfigs.DPI)
        divider = make_axes_locatable(ax_heatmap)
        ax_colorbar = divider.append_axes("top", 0.2, pad=0.2)
        # cluster rows
        if FigsConfigs.CLUSTER_PCA_ITEM_ROWS:
            lnk0 = linkage(pdist(item_pca_mat))
            dg0 = dendrogram(lnk0,
                             no_plot=True)
            z = item_pca_mat[dg0['leaves'], :]
            yticklabels = np.array(pca_item_list)[dg0['leaves']]
        else:
            z = item_pca_mat
            yticklabels = pca_item_list
        # heatmap
        max_extent = ax_heatmap.get_ylim()[1]
        vmin, vmax = round(np.min(z), 1), round(np.max(z), 1)
        im = ax_heatmap.imshow(z[::-1], aspect='auto',
                               cmap=plt.cm.jet,
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='horizontal')
        cb.ax.set_xticklabels([vmin, vmax], rotation=0, fontsize=FigsConfigs.LEG_FONTSIZE)
        cb.set_label('Loading', labelpad=-40, rotation=0, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # set heatmap ticklabels
        xlim = ax_heatmap.get_xlim()[1]
        halfxw = 0.5 * xlim / FigsConfigs.NUM_PCS
        ax_heatmap.xaxis.set_ticks(np.linspace(halfxw, xlim - halfxw, FigsConfigs.NUM_PCS))
        if label_expl_var:
            ax_heatmap.xaxis.set_ticklabels(['PC {} ({:.1f} %)'.format(pc_id + 1, expl_var)
                                             for pc_id, expl_var in zip(range(FigsConfigs.NUM_PCS), expl_vars)])
        else:
            ax_heatmap.xaxis.set_ticklabels(['PC {}'.format(pc_id + 1)
                                             for pc_id in range(FigsConfigs.NUM_PCS)])
        ylim = ax_heatmap.get_ylim()[1]
        halfyw = 0.5 * ylim / num_items
        ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, num_items))
        ax_heatmap.yaxis.set_ticklabels(yticklabels)
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # set label rotation and fontsize
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        fig.subplots_adjust(bottom=0.2)  # make room for tick labels
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_principal_comps_table_fig(pc_id):
        """
        Returns fig showing tokens along principal component axis
        """
        start = time.time()

        num_cols = 2
        col_labels = ['Low End', 'High End']
        # load data
        acts_mat = model.term_acts_mat if FigsConfigs.SVD_IS_TERMS else model.get_multi_probe_prototype_acts_df().values
        tokens = model.hub.train_terms.types if FigsConfigs.SVD_IS_TERMS else model.hub.probe_store.types
        pca_model = sklearnPCA(n_components=FigsConfigs.NUM_PCS)
        pcs = pca_model.fit_transform(acts_mat)
        expl_var_perc = np.asarray(pca_model.explained_variance_ratio_) * 100
        pc = pcs[:, pc_id]
        # sort and filter
        sorted_pc, sorted_tokens = list(zip(*sorted(zip(pc, tokens), key=itemgetter(0))))
        col0_strs = ['{} {:.2f} (freq: {:,})'.format(term, loading, sum(model.hub.term_part_freq_dict[term]))
                     for term, loading in zip(
                sorted_tokens[:FigsConfigs.NUM_PCA_LOADINGS], sorted_pc[:FigsConfigs.NUM_PCA_LOADINGS])
                     if sum(model.hub.term_part_freq_dict[term]) > FigsConfigs.PCA_FREQ_THR]
        col1_strs = ['{} {:.2f} ({:,})'.format(term, loading, sum(model.hub.term_part_freq_dict[term]))
                     for term, loading in zip(
                sorted_tokens[-FigsConfigs.NUM_PCA_LOADINGS:][::-1], sorted_pc[-FigsConfigs.NUM_PCA_LOADINGS:][::-1])
                     if sum(model.hub.term_part_freq_dict[term]) > FigsConfigs.PCA_FREQ_THR]
        # make probes_mat
        max_rows = max(len(col0_strs), len(col1_strs))
        probes_mat = np.chararray((max_rows, num_cols), itemsize=40, unicode=True)
        probes_mat[:] = ''  # initialize so that mpl can read table
        probes_mat[:len(col0_strs), 0] = col0_strs
        probes_mat[:len(col1_strs), 1] = col1_strs
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 0.25 * max_rows), dpi=FigsConfigs.DPI)
        ax.set_title('Principal Component {} ({:.2f}% var)'.format(
            pc_id, expl_var_perc[pc_id]), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.axis('off')
        # plot
        table_ = ax.table(cellText=probes_mat, colLabels=col_labels,
                          loc='center', colWidths=[0.3] * num_cols)
        table_.auto_set_font_size(False)
        table_.set_fontsize(8)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_principal_comps_line_fig(pc_id):
        """
        Returns fig showing loadings on a specified principal component
        """
        start = time.time()

        # load data
        acts_mat = model.term_acts_mat if FigsConfigs.SVD_IS_TERMS else model.get_multi_probe_prototype_acts_df().values
        pca_model = sklearnPCA(n_components=FigsConfigs.NUM_PCS)
        pcs = pca_model.fit_transform(acts_mat)
        pc = pcs[:, pc_id]
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        # plot
        ax.plot(sorted(pc), '-', linewidth=FigsConfigs.LINEWIDTH, color='black')
        ax.set_xticklabels([])
        ax.set_xlabel('Token IDs', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('PC {} Loading'.format(pc_id), fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_scree_fig():
        """
        Returns fig showing amount of variance accounted for by each principal component
        """
        start = time.time()

        # load data
        acts_mat = model.term_acts_mat if FigsConfigs.SVD_IS_TERMS else model.get_multi_probe_prototype_acts_df().values
        pca_model = sklearnPCA(n_components=acts_mat.shape[1])
        pca_model.fit(acts_mat)
        expl_var_perc = np.asarray(pca_model.explained_variance_ratio_) * 100
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.set_xticklabels([])
        ax.set_xlabel('Principal Component', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('% Var Explained (Cumulative)', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylim([0, 100])
        # plot
        ax.plot(expl_var_perc.cumsum(), '-', linewidth=FigsConfigs.LINEWIDTH, color='black')
        ax.plot(expl_var_perc.cumsum()[:FigsConfigs.NUM_PCS], 'o', linewidth=FigsConfigs.LINEWIDTH, color='black')
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig
