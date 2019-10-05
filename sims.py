    def make_probe_sim_hist_fig(num_bins=1000):  # TODO make this for all tokens not just probes
        """
        Returns fig showing histogram of similarities learned by "model"
        """
        start = time.time()

        # load data
        probes_acts_df = model.get_multi_probe_prototype_acts_df()
        probe_simmat = cosine_similarity(probes_acts_df.values)
        probe_simmat[np.tril_indices(probe_simmat.shape[0], -1)] = np.nan
        probe_simmat_values = probe_simmat[~np.isnan(probe_simmat)]
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        ax.set_xlabel('Similarity', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_ylabel('Frequency', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.xaxis.grid(True)
        # plot
        step_size = 1.0 / num_bins
        bins = np.arange(-1, 1, step_size)
        hist, _ = np.histogram(probe_simmat_values, bins=bins)
        x_binned = bins[:-1]
        ax.plot(x_binned, hist, '-', linewidth=FigsConfigs.LINEWIDTH, c='black')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_compare_term_simmats_fig(num_most_frequent=10,
                                      dists=(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)):
        """
        Returns fig showing similarity between a computed similarity space and that of the model
        """
        start = time.time()

        most_frequent_terms = model.hub.get_most_frequent_terms(num_most_frequent)
        # load data
        y = []
        for dist in dists:
            term_context_mat = np.zeros((model.hub.train_terms.num_types, num_most_frequent))
            for n, term in enumerate(model.hub.train_terms.types):
                terms_near_term = model.hub.get_terms_near_term(term, dist)
                context_vec = [terms_near_term.count(term) for term in most_frequent_terms]
                term_context_mat[n] = context_vec
            term_context_simmat = cosine_similarity(term_context_mat)
            # calc fit
            fit = paired_cosine_distances(term_context_simmat, model.term_simmat).mean()
            y.append(fit)
        x = np.asarray(dists)
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, 3), dpi=FigsConfigs.DPI)
        plt.title('Terms')
        ax.set_ylabel('Fit', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.set_xlabel('Context Distance', fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top='off', right='off')
        ax.axhline(y=0, color='grey', zorder=0)
        ax.set_xticks(dists)
        ax.set_xticklabels(dists)
        # plot
        width = 0.3
        ax.bar(x, y, width, color='black')
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig
