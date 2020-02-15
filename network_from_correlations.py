from typing import List
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def make_network_from_correlations_fig(data_matrix: np.ndarray,
                                       words: List[str],
                                       threshold: float,
                                       ) -> plt.Figure:

    # make links and threshold
    corr = pd.DataFrame(data=data_matrix, index=words, columns=words)
    links = corr.stack().reset_index()
    links.columns = ['word1', 'word2', 'correlation']
    links_filtered = links.loc[(links['correlation'] > threshold) & (links['word1'] != links['word2'])]

    # fig
    res, ax = plt.subplots(dpi=163)
    plt.title('link threshold={}'.format(threshold))
    G = nx.DiGraph()
    G.add_weighted_edges_from([tuple(link) for link in links_filtered.values])
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G,
            ax=ax,
            pos=pos,
            with_labels=True,
            node_color='orange',
            node_size=200,
            edge_color=links_filtered['correlation'].tolist(),
            edge_cmap=plt.cm.Blues,
            linewidths=1,
            font_size=8)

    return res


DATA_SIZE = 32
THRESHOLD = 0.5

# make random correlation matrix
data_matrix = np.random.random((DATA_SIZE, DATA_SIZE))
words = [f'w{i}' for i in range(DATA_SIZE)]

fig = make_network_from_correlations_fig(data_matrix, words, THRESHOLD)
fig.show()
