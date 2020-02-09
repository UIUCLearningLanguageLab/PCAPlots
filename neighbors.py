def make_neighbors_figs(model, field_input):
    def make_neighbors_table_fig(terms):
        """
        Returns fig showing 10 nearest neighbors from "model" for "tokens"
        """
        start = time.time()
        # load data
        neighbors_mat_list = []
        col_labels_list = []
        num_cols = 5  # fixed
        num_neighbors = 10  # fixed
        for i in range(0, len(terms), num_cols):  # split probes into even sized lists
            col_labels = terms[i:i + num_cols]
            neighbors_mat = np.chararray((num_neighbors, num_cols), itemsize=20, unicode=True)
            neighbors_mat[:] = ''  # initialize so that mpl can read table
            # make column
            for col_id, term in enumerate(col_labels):
                term_id = model.hub.train_terms.term_id_dict[term]
                token_sims = model.term_simmat[term_id]
                neighbor_tuples = [(model.hub.train_terms.types[term_id], token_sim) for term_id, token_sim in
                                   enumerate(token_sims)
                                   if model.hub.train_terms.types[term_id] != term]
                neighbor_tuples = sorted(neighbor_tuples, key=itemgetter(1), reverse=True)[:num_neighbors]
                neighbors_mat_col = ['{:>15} {:.2f}'.format(tuple[0], tuple[1])
                                     for tuple in neighbor_tuples if tuple[0] != term]
                neighbors_mat[:, col_id] = neighbors_mat_col
            # collect info for plotting
            neighbors_mat_list.append(neighbors_mat)
            length_diff = num_cols - len(col_labels)
            col_labels_list.append(col_labels + [' '] * length_diff)
        # fig
        num_tables = max(2, len(neighbors_mat_list))  # max 2 otherwise axarr is  not indexable
        fig, axarr = plt.subplots(num_tables, 1,
                                  figsize=(config.Fig.MAX_FIG_WIDTH, num_tables * (num_neighbors / 4.)),
                                  dpi=config.Fig.DPI)
        for ax, neighbors_mat, col_labels in zip_longest(axarr, neighbors_mat_list, col_labels_list):
            ax.axis('off')
            if neighbors_mat is not None:  # this allows turning off of axis even when neighbors_mat list length is < 2
                table_ = ax.table(cellText=neighbors_mat, colLabels=col_labels,
                                  loc='center', colWidths=[0.2] * num_cols)
                table_.auto_set_font_size(False)
                table_.set_fontsize(8)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    def make_sg_neighbors_table_fig(terms, model_id=config.Fig.SKIPGRAM_MODEL_ID):
        """
        Returns fig showing nearest 10 neighbors from pre-trained skip-gram model for "tokens"
        """
        start = time.time()

        # load data
        token_simmat = np.load(GlobalConfigs.SG_DIR / 'sg_token_simmat_model_{}.npy'.format(model_id))
        token_list = np.load(GlobalConfigs.SG_DIR, 'sg_types.npy')
        token_id_dict = {token: token_id for token_id, token in enumerate(token_list)}
        # make neighbors_mat
        neighbors_mat_list = []
        col_labels_list = []
        num_cols = 5  # fixed
        num_neighbors = 10  # fixed
        for i in range(0, len(terms), num_cols):  # split probes into even sized lists
            col_labels = terms[i:i + num_cols]
            neighbors_mat = np.chararray((num_neighbors, num_cols), itemsize=20, unicode=True)
            neighbors_mat[:] = ''  # initialize so that mpl can read table
            # make column
            for col_id, token in enumerate(col_labels):
                token_id = token_id_dict[token]
                token_sims = token_simmat[token_id]
                neighbor_tuples = [(token_list[token_id], token_sim) for token_id, token_sim in enumerate(token_sims)
                                   if token_list[token_id] != token]
                neighbor_tuples = sorted(neighbor_tuples, key=itemgetter(1), reverse=True)[:num_neighbors]
                neighbors_mat_col = ['{:>15} {:.2f}'.format(tuple[0], tuple[1])
                                     for tuple in neighbor_tuples if tuple[0] != token]
                neighbors_mat[:, col_id] = neighbors_mat_col
            # collect info for plotting
            neighbors_mat_list.append(neighbors_mat)
            length_diff = num_cols - len(col_labels)
            col_labels_list.append(['{} (skip-gram)'.format(token_) for token_ in col_labels] + [' '] * length_diff)
        # fig
        num_tables = max(2, len(neighbors_mat_list))  # max 2 otherwise axarr is  not indexable
        fig, axarr = plt.subplots(num_tables, 1,
                                  figsize=(config.Fig.MAX_FIG_WIDTH, num_tables * (num_neighbors / 4.)),
                                  dpi=config.Fig.DPI)
        for ax, neighbors_mat, col_labels in zip_longest(axarr, neighbors_mat_list, col_labels_list):
            ax.axis('off')
            if neighbors_mat is not None:  # this allows turning off of axis even when neighbors_mat list length is < 2
                table_ = ax.table(cellText=neighbors_mat, colLabels=col_labels,
                                  loc='center', colWidths=[0.2] * num_cols)
                table_.auto_set_font_size(False)
                table_.set_fontsize(8)
        fig.tight_layout()
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig

    # figs = [make_neighbors_table_fig(field_input),
    #         make_sg_neighbors_table_fig(field_input)]  # TODO implement sg
    figs = [make_neighbors_table_fig(field_input)]
    return figs
