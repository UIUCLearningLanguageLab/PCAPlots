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

    def make_sim_simmat_fig(include_sg=False, vmin=0.9, config_id=1):
        """
        Returns fig showing similarity matrix of probe similarity matrices of multiple models
        """
        start = time.time()
        # init term_simmat_mat
        if include_sg:
            sg_types = list(np.load(os.path.join(GlobalConfigs.SG_DIR, 'sg_types.npy')).tolist())
            terms = list(set(model_groups[0][0].hub.train_terms.types) & set(sg_types))
            dim1 = model_groups[0][0].hub.probe_store.num_probes * len(terms)
        else:
            sg_types = None
            terms = model_groups[0][0].hub.train_terms.types
            dim1 = model_groups[0][0].hub.probe_store.num_probes * len(terms)
        num_models = len([model for models in model_groups for model in models])
        term_simmat_mat = np.zeros((num_models, dim1))
        # load data
        row_ids = iter(range(num_models))
        probe_term_ids = [model_groups[0][0].hub.train_terms.types.index(probe_)
                          for probe_ in model_groups[0][0].hub.probe_store.types]
        term_term_ids = [model_groups[0][0].hub.train_terms.types.index(term)
                         for term in terms]
        group_names = []
        for model_desc, models in zip(model_descs, model_groups):
            for model in models:
                group_name = model_desc.split('\n')[config_id].split('=')[-1]
                group_names.append(group_name)
                probe_term_simmat = model.term_simmat[probe_term_ids, :]
                flattened = probe_term_simmat[:, term_term_ids, ].flatten()
                term_simmat_mat[next(row_ids), :] = flattened
        # sim_simmat
        if include_sg:  # TODO fix
            sg_probe_term_ids = [sg_types.index(probe) for probe in model_groups[0][0].hub.probe_store.types]
            sg_term_term_ids = [sg_types.index(term) for term in terms]
            sg_token_simmat_filenames = sorted([f for f in os.listdir(GlobalConfigs.SG_DIR)
                                                if f.startswith('sg_token_simmat')])
            num_sgs = len(sg_token_simmat_filenames)
            sg_term_simmats_mat = np.zeros((num_sgs, dim1))
            for model_id, f in enumerate(sg_token_simmat_filenames):
                sg_probe_term_simmat = np.load(os.path.join(GlobalConfigs.SG_DIR, f))[sg_probe_term_ids, :]
                flattened = sg_probe_term_simmat[:, sg_term_term_ids].flatten()
                sg_term_simmats_mat[model_id, :] = flattened
            group_names += ['skip-gram'] * num_sgs
            term_simmat_mat = np.vstack((term_simmat_mat, sg_term_simmats_mat))
        else:
            term_simmat_mat = term_simmat_mat
        sim_simmat = pd.DataFrame(term_simmat_mat).T.corr().values
        # fig
        fig, ax = plt.subplots(figsize=(FigsConfigs.MAX_FIG_WIDTH, FigsConfigs.MAX_FIG_WIDTH))
        mask = np.zeros_like(sim_simmat, dtype=np.bool)
        mask[np.triu_indices_from(mask, 1)] = True
        sns.heatmap(sim_simmat, ax=ax, square=True, annot=False,
                    annot_kws={"size": 5}, cbar_kws={"shrink": .5},
                    vmin=vmin, vmax=1.0, cmap='jet')  # , mask=mask
        # colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([vmin, 1.0])
        cbar.set_ticklabels([str(vmin), '1.0'])
        cbar.set_label('Similarity between Semantic Spaces')
        # ax (needs to be below plot for axes to be labeled)
        num_group_names = len(group_names)
        ax.set_yticks(np.arange(num_group_names) + 0.5)
        ax.set_xticks(np.arange(num_group_names) + 0.5)
        ax.set_yticklabels(group_names, rotation=0)
        ax.set_xticklabels(group_names, rotation=90)
        plt.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        # export to csv for stats  # TODO put this somewhere else
        if AppConfigs.SAVE_SIM_SIMMAT:
            col1 = ['0'] * num_models1 * num_models + ['1'] * num_models2 * num_models
            col2 = (['0'] * num_models1 + ['1'] * num_models2) * num_models
            sim_simmat[np.tril_indices(sim_simmat.shape[0], 0)] = np.nan
            col3 = sim_simmat.flatten()
            sim_simmat_df = pd.DataFrame(data={'model1': col1,
                                               'model2': col2,
                                               'sim': col3})
            sim_simmat_df = sim_simmat_df.dropna()
            sim_simmat_df.to_csv(os.path.join(os.path.expanduser("~"), 'sim_simmat_df.csv'), index=False)
            print('Saved sim_simmat_df.csv to {}'.format(dir))
        return fig
