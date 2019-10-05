        def make_cat_sim_dh_fig(num_colors=None, y_title=False, vmin=0.0, vmax=1.0):
        """
        Returns fig showing dendrogram heatmap of category similarity matrix
        """
        start = time.time()

        # load data
        cat_simmat = calc_cat_sim_mat(model)
        cat_simmat_labels = model.hub.probe_store.cats
        print('Cat simmat  min: {} max {}'.format(np.min(cat_simmat), np.max(cat_simmat)))
        print('Fig  min: {} max {}'.format(vmin, vmax))
        # fig
        fig, ax_heatmap = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 7), dpi=FigsConfigs.DPI)
        ax_heatmap.yaxis.tick_right()
        divider = make_axes_locatable(ax_heatmap)
        ax_denright = divider.append_axes("right", 0.8, pad=0.0, sharey=ax_heatmap)
        ax_denright.set_frame_on(False)
        ax_colorbar = divider.append_axes("top", 0.1, pad=0.4)
        # dendrogram
        lnk0 = linkage(pdist(cat_simmat))
        if num_colors is None or num_colors <= 1:
            left_threshold = -1
        else:
            left_threshold = 0.5 * (lnk0[1 - num_colors, 2] +
                                    lnk0[-num_colors, 2])
        dg0 = dendrogram(lnk0, ax=ax_denright,
                         orientation='right',
                         color_threshold=left_threshold,
                         no_labels=True)
        # Reorder the values in x to match the order of the leaves of the dendrograms
        z = cat_simmat[dg0['leaves'], :]  # sorting rows
        z = z[:, dg0['leaves']]  # sorting columns for symmetry
        # heatmap
        max_extent = ax_denright.get_ylim()[1]
        im = ax_heatmap.imshow(z[::-1], aspect='auto',
                               cmap=plt.cm.jet,
                               extent=(0, max_extent, 0, max_extent),
                               vmin=vmin, vmax=vmax)
        # colorbar
        cb = plt.colorbar(im, cax=ax_colorbar, ticks=[vmin, vmax], orientation='horizontal')
        cb.ax.set_xticklabels([vmin, vmax], fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        cb.set_label('Correlation Coefficient', labelpad=-10, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # set heatmap ticklabels
        xlim = ax_heatmap.get_xlim()[1]
        ncols = len(cat_simmat_labels)
        halfxw = 0.5 * xlim / ncols
        ax_heatmap.xaxis.set_ticks(np.linspace(halfxw, xlim - halfxw, ncols))
        ax_heatmap.xaxis.set_ticklabels(np.array(cat_simmat_labels)[dg0['leaves']])  # for symmetry
        ylim = ax_heatmap.get_ylim()[1]
        nrows = len(cat_simmat_labels)
        halfyw = 0.5 * ylim / nrows
        if y_title:
            ax_heatmap.yaxis.set_ticks(np.linspace(halfyw, ylim - halfyw, nrows))
            ax_heatmap.yaxis.set_ticklabels(np.array(cat_simmat_labels)[dg0['leaves']])
        # Hide all tick lines
        lines = (ax_heatmap.xaxis.get_ticklines() +
                 ax_heatmap.yaxis.get_ticklines() +
                 ax_denright.xaxis.get_ticklines() +
                 ax_denright.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
        # set label rotation and fontsize
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # make dendrogram labels invisible
        plt.setp(ax_denright.get_yticklabels() + ax_denright.get_xticklabels(),
                 visible=False)
        fig.subplots_adjust(bottom=0.2)  # make room for tick labels
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig
    
    
    def make_probe_sim_dh_fig_jw(probes,
                                 num_colors=None,
                                 label_doc_id=True,
                                 vmin=0.90, vmax=1.0,
                                 cluster=False):
        """
        Returns fig showing dendrogram heatmap of similarity matrix of "probes"
        """
        start = time.time()

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
        fig, ax_heatmap = plt.subplots(figsize=(11, 7), dpi=FigsConfigs.DPI)
        if label_doc_id:
            plt.title('Trained on {:,} terms'.format(model.mb_size * int(model.mb_name)),
                      fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
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
        ax_freqs.set_xlim([0, FigsConfigs.PROBE_FREQ_YLIM])
        ax_freqs.set_xticks([0, FigsConfigs.PROBE_FREQ_YLIM])
        ax_freqs.set_xticklabels([0, FigsConfigs.PROBE_FREQ_YLIM])
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
        cb.ax.set_xticklabels([vmin, vmax], fontsize=FigsConfigs.AXLABEL_FONT_SIZE, rotation=90)
        cb.set_label('Correlation Coefficient', labelpad=-10, fontsize=FigsConfigs.AXLABEL_FONT_SIZE, rotation=90)
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
        xlbls = ax_heatmap.xaxis.get_ticklabels()
        plt.setp(xlbls, rotation=-90)
        plt.setp(xlbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ylbls = ax_heatmap.yaxis.get_ticklabels()
        plt.setp(ylbls, rotation=0)
        plt.setp(ylbls, fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        # make dendrogram labels invisible
        plt.setp(ax_dend.get_yticklabels() + ax_dend.get_xticklabels(),
                 visible=False)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig
