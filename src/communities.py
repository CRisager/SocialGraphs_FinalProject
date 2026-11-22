"""
Community detection (Step 4): Louvain modularity maximisation on the backbone LCCs.

Louvain (Blondel et al. 2008, nx.community.louvain_communities) is the course's
standard method: it is fast, uses the edge weights, and gives a modularity M that the
null models of Step 6 can be compared with. It is randomised (the order in which nodes
are visited), so every call takes the seed explicitly; Step 6C checks how much the
partition depends on it.

Communities are numbered by size (0 = largest), so the numbering is the same whatever
order Louvain returns them in, and stored on the nodes as the attribute "community".
"""

import networkx as nx
import numpy as np
import pandas as pd

from .backbone import canonical


def louvain(G, seed, weight="weight", resolution=1.0):
    """Louvain communities of G as a list of sets, largest first. G is put in canonical
    (sorted) order first, so a seed gives the same partition in every run."""
    comms = nx.community.louvain_communities(canonical(G), weight=weight,
                                             resolution=resolution, seed=seed)
    return sorted(comms, key=lambda c: (-len(c), min(map(str, c))))


def membership(comms):
    """{node: community number} for a list of sets."""
    return {n: i for i, c in enumerate(comms) for n in c}


def detect(G, seed, weight="weight"):
    """Run Louvain, store the partition on the nodes ("community") and return
    (communities, modularity M)."""
    comms = louvain(G, seed, weight)
    nx.set_node_attributes(G, membership(comms), "community")
    return comms, nx.community.modularity(G, comms, weight=weight)


def size_summary(comms, min_size=10):
    """Number of communities, how many have at least min_size songs and the share of
    songs in those, and the size distribution."""
    sizes = np.array(sorted(map(len, comms), reverse=True))
    big = sizes[sizes >= min_size]
    return {"communities": len(sizes), f"communities_ge{min_size}": len(big),
            f"share_in_ge{min_size}": big.sum() / sizes.sum(),
            "largest": int(sizes[0]), "largest_share": sizes[0] / sizes.sum(),
            "median_size": float(np.median(sizes)), "singletons": int((sizes == 1).sum())}


def composition(G, attr, comms=None, min_size=10, normalize=True):
    """Community x label table for node attribute `attr` (e.g. genre_eval), rows =
    communities with at least min_size nodes, nodes without the label left out.
    With normalize, rows are percentages."""
    comm = nx.get_node_attributes(G, "community")
    lab = nx.get_node_attributes(G, attr)
    df = pd.DataFrame({"community": pd.Series(comm), "label": pd.Series(lab)}).dropna()
    sizes = pd.Series(comm).value_counts()
    df = df[df.community.isin(sizes[sizes >= min_size].index)]
    table = pd.crosstab(df.community, df.label)
    return table.div(table.sum(axis=1), axis=0).mul(100) if normalize else table


# toy check: two triangles joined by one edge -> two communities, largest first
_G = nx.Graph([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3), (5, 6)])
nx.set_edge_attributes(_G, 1.0, "weight")
_c, _m = detect(_G, seed=0)
assert len(_c) == 2 and len(_c[0]) == 4 and _G.nodes[6]["community"] == 0 and _m > 0.3
