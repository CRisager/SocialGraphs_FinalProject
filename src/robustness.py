"""
Robustness of the communities (Step 6C).

Three questions, each answered by re-running Louvain and comparing:

1. Seeds. Louvain is randomised, so one partition could be a lucky draw. We run it
   with 20 seeds on the same backbone LCC and report the mean AMI over all pairs of
   partitions (1 = always the same partition), plus the spread of M, the number of
   communities and the AMI with genre and country over the seeds.
2. Alpha. The backbone depends on the disparity-filter level alpha. We rebuild each
   backbone at alpha = 0.01, 0.05 and 0.1 (a cheap threshold on the stored p-values),
   take the LCC, run Louvain with SEED and report size, M, communities and AMI.
3. Country weight (G_combined only). Country dominates G_combined's communities
   (Step 5). Rebuilding it with the country projection scaled down shows how much of
   the genre signal is hidden by country.
"""

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

from .backbone import backbone, largest_component
from .communities import louvain, membership


def ami_with(G, member, attr):
    """AMI between a partition ({node: community}) and node attribute `attr`, on the
    nodes that have the label."""
    lab = pd.Series(nx.get_node_attributes(G, attr)).dropna()
    com = pd.Series(member).reindex(lab.index)
    return adjusted_mutual_info_score(lab, com)


def partition_row(G, comms, attrs=(("genre_eval", "genre"), ("country", "country"))):
    """Size, M, number of communities and AMI with each label for one partition."""
    member = membership(comms)
    row = {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
           "M": nx.community.modularity(G, comms, weight="weight"),
           "n_communities": len(comms), "n_communities_ge10": sum(len(c) >= 10 for c in comms)}
    row.update({f"AMI_{key}": ami_with(G, member, attr) for attr, key in attrs})
    return row


def seed_runs(G, seeds):
    """(DataFrame with one partition_row per seed, list of memberships)."""
    rows, members = [], []
    for s in seeds:
        comms = louvain(G, s)
        rows.append({"seed": s, **partition_row(G, comms)})
        members.append(membership(comms))
    return pd.DataFrame(rows), members


def pairwise_ami(members):
    """AMI for every pair of partitions of the same nodes."""
    nodes = sorted(members[0], key=str)
    labels = [[m[n] for n in nodes] for m in members]
    return np.array([adjusted_mutual_info_score(a, b) for a, b in combinations(labels, 2)])


def alpha_run(P, alpha, seed):
    """Backbone of the p-value graph P at alpha -> LCC -> Louvain -> partition_row."""
    g = largest_component(backbone(P, alpha))
    return partition_row(g, louvain(g, seed))


# toy checks: identical partitions agree perfectly, and a two-triangle graph is split
# the same way by every seed
_G = nx.Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
nx.set_edge_attributes(_G, 1.0, "weight")
nx.set_node_attributes(_G, {n: "ab"[n // 3] for n in _G}, "genre_eval")
nx.set_node_attributes(_G, {n: "x" for n in _G}, "country")
_df, _m = seed_runs(_G, range(5))
assert np.allclose(pairwise_ami(_m), 1.0) and np.allclose(_df.AMI_genre, 1.0)
