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
