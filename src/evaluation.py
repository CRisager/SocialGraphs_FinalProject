"""
Communities vs metadata (Step 5): genre and country.

Genre and country are never used to build the graphs; here they are only compared
with the Louvain partitions stored on the nodes ("community"). Every measure uses only
the songs that have the label.

  AMI / ARI     adjusted mutual information and adjusted Rand index between
                community and label (sklearn). Both are 0 for agreement at chance
                level and 1 for identical partitions; AMI is about shared information,
                ARI about pairs of songs put together. Plain NMI is not used: it is
                above 0 even for random partitions.
  assortativity nx.attribute_assortativity_coefficient on the labelled songs: do
                links join songs with the same label more often than chance? It needs
                no communities, so it does not depend on Louvain. It ignores weights.
  M_label       the modularity of the label itself used as the partition, on the
                subgraph of labelled songs, next to the modularity of the Louvain
                partition restricted to the same songs. Songs without the label are
                left out (rather than put in one "unknown" group, which would act as an
                extra genre, or in groups of their own, which would lower M_label for
                reasons unrelated to genre).
"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


def labelled(G, attr, nodes=None):
    """DataFrame (index = node) of community and label for the nodes that have both,
    optionally restricted to `nodes`."""
    df = pd.DataFrame({"community": pd.Series(nx.get_node_attributes(G, "community")),
                       "label": pd.Series(nx.get_node_attributes(G, attr))})
    df = df[df.label.notna() & df.community.notna()]
    if nodes is not None:
        df = df[df.index.isin(set(nodes))]
    return df


def agreement(G, attr, nodes=None):
    """AMI, ARI and the number of songs compared."""
    df = labelled(G, attr, nodes)
    return {"n": len(df), "AMI": adjusted_mutual_info_score(df.label, df.community),
            "ARI": adjusted_rand_score(df.label, df.community)}


def assortativity(G, attr):
    """Attribute assortativity on the subgraph of songs with the label."""
    keep = [n for n, v in G.nodes(data=attr) if isinstance(v, str)]
    return nx.attribute_assortativity_coefficient(G.subgraph(keep), attr)


def label_modularity(G, attr, weight="weight"):
    """(M of the label partition, M of the Louvain partition, number of songs), both on
    the subgraph of songs with the label."""
    df = labelled(G, attr)
    H = G.subgraph(df.index)
    by = lambda col: [set(g.index) for _, g in df.groupby(col)]
    return (nx.community.modularity(H, by("label"), weight=weight),
            nx.community.modularity(H, by("community"), weight=weight), H.number_of_nodes())


def top_labels(values, n, other="other"):
    """Keep the n most common labels, others -> `other` (for readable heatmaps)."""
    s = pd.Series(values)
    keep = s.value_counts().index[:n]
    return s.where(s.isin(keep) | s.isna(), other)


# toy check: two triangles, perfectly split by both community and label
_G = nx.Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
nx.set_edge_attributes(_G, 1.0, "weight")
nx.set_node_attributes(_G, {n: n // 3 for n in _G}, "community")
nx.set_node_attributes(_G, {n: "ab"[n // 3] for n in _G}, "g")
assert agreement(_G, "g")["AMI"] == 1.0 and agreement(_G, "g")["ARI"] == 1.0
_ml, _mc, _n = label_modularity(_G, "g")
assert np.isclose(_ml, _mc) and _n == 6 and assortativity(_G, "g") > 0.7
