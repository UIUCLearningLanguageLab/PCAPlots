import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects
from math import gcd

from src import config


def equidistant_elements(l, n):
    """return "n" elements from "l" such that they are equally far apart iin "l" """
    while not gcd(n, len(l)) == n:
        l.pop()
    step = len(l) // n
    ids = np.arange(step, len(l) + step, step) - 1  # -1 for indexing
    res = np.asarray(l)[ids].tolist()
    return res


def make_cat_acts_2d_walk_fig(component1: int,
                              component2: int,
                              ) -> plt.Figure:
    """
    Returns fig showing evolution of average category activations in 2D space using PCA.
    """

    palette = np.array(sns.color_palette("hls", model.hub.probe_store.num_cats))
    num_saved_mb_names = len(model.ckpt_mb_names)
    num_walk_timepoints = min(num_saved_mb_names, config.Fig.DEFAULT_NUM_WALK_TIMEPOINTS)
    walk_mb_names = equidistant_elements(model.ckpt_mb_names, num_walk_timepoints)

    # fit pca model on last data_step
    num_components = component2
    pca_model = PCA(n_components=num_components)
    cat_prototype_acts_df = model.get_multi_cat_prototype_acts_df()
    pca_model.fit(cat_prototype_acts_df.values)

    # transform acts from remaining data_steps with pca model
    cat_acts_2d_mats = []
    for mb_name in walk_mb_names:
        cat_prototype_acts_df = model.get_multi_cat_prototype_acts_df()
        cat_act_2d_pca = pca_model.transform(cat_prototype_acts_df.values)
        cat_acts_2d_mats.append(cat_act_2d_pca[:, num_components - 2:])

    # fig
    fig, ax = plt.subplots(figsize=(config.Fig.MAX_FIG_WIDTH, 7), dpi=config.Fig.DPI)
    for cat_id, cat in enumerate(model.hub.probe_store.cats):
        x, y = zip(*[acts_2d_mat[cat_id] for acts_2d_mat in cat_acts_2d_mats])
        ax.plot(x, y, c=palette[cat_id], lw=config.Fig.LINEWIDTH)
        xtext, ytext = cat_acts_2d_mats[-1][cat_id, :]
        txt = ax.text(xtext, ytext, str(cat), fontsize=8,
                      color=palette[cat_id])
        txt.set_path_effects([
            patheffects.Stroke(linewidth=config.Fig.LINEWIDTH, foreground="w"), patheffects.Normal()])
    ax.axis('off')
    x_maxval = np.max(np.dstack(cat_acts_2d_mats)[:, 0, :]) * 1.2
    y_maxval = np.max(np.dstack(cat_acts_2d_mats)[:, 1, :]) * 1.2
    ax.set_xlim([-x_maxval, x_maxval])
    ax.set_ylim([-y_maxval, y_maxval])
    ax.axhline(y=0, linestyle='--', c='grey', linewidth=1.0)
    ax.axvline(x=0, linestyle='--', c='grey', linewidth=1.0)
    ax.set_title(f'Principal components {component1} and {component2}\nEvolution across training')
    fig.tight_layout()

    return fig
