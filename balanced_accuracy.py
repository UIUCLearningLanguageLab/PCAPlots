    def make_ba_breakdown_annotated_fig(context_type):
        """
        Returns fig showing ranking of each probe's avg_probe_fs broken down by category
        """
        start = time.time()
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
        fig, axarr = plt.subplots(num_axes, figsize=(FigsConfigs.MAX_FIG_WIDTH, 6 * num_axes), dpi=FigsConfigs.DPI)
        for n, ax in enumerate(axarr):
            # truncate data
            xys_truncated = xys[n * cats_per_axis: (n + 1) * cats_per_axis]
            cats_sorted_by_fs_truncated = cats_sorted_by_fs[n * cats_per_axis: (n + 1) * cats_per_axis]
            # axis
            ax.set_ylabel('Avg Probe Balanced Accuracy ({})'.format(context_type),
                          fontsize=FigsConfigs.AXLABEL_FONT_SIZE)
            ax.set_xticks(np.arange(cats_per_axis), minor=False)  # shifts xtick labels right
            ax.set_xticklabels(cats_sorted_by_fs_truncated, minor=False, fontsize=FigsConfigs.TICKLABEL_FONT_SIZE,
                               rotation=90)
            ax.set_xlim([0, cats_per_axis])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='both', top='off', right='off')
            ax.axhline(y=mean_fs, alpha=FigsConfigs.FILL_ALPHA, c='grey', linestyle='--', zorder=1)
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
        print('{} completed in {:.1f} secs'.format(sys._getframe().f_code.co_name, time.time() - start))
        return fig
