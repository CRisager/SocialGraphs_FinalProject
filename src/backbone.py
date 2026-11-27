"""
Disparity-filter backbone (Serrano, Boguna & Vespignani 2009, PNAS 106:6483) - Step 2.

Why: the song graphs are dense (the projections are >90% complete) and their weights
are broadly distributed, so a global weight threshold would keep all edges of the
strong nodes and none of the weak ones. The disparity filter instead asks, for each
node, which of its edges carry more of its strength than expected if the strength
were spread uniformly at random over its k edges.

For an edge (u, v) seen from u, with s_u = u's strength and k_u = u's degree:
    p_u(u, v) = (1 - w_uv / s_u) ** (k_u - 1)
the probability that a uniform split of s_u gives one edge at least w_uv. The edge is
kept if p < alpha from EITHER endpoint, so it is enough that it matters to one of the
two songs. A node of degree 1 has p = 0**0 = 1 by the formula, but its single edge
carries all its strength, so that edge is always kept (Serrano et al. do the same).

disparity_pvalues stores p = min(p_u, p_v) on every edge once; backbone(G, alpha) is
then a cheap threshold, which makes the alpha sensitivity check (Step 6C) fast.
"""

import networkx as nx
import numpy as np


def disparity_pvalues(G, weight="weight"):
    """Copy of G with edge attribute 'p': the smaller of the two endpoints' disparity
    p-values (0 for an edge of a degree-1 node)."""
    H = G.copy()
    if H.number_of_edges() == 0:
        return H
    strength = dict(H.degree(weight=weight))
    degree = dict(H.degree())
    u, v, w = zip(*H.edges(data=weight))
    w = np.asarray(w, dtype=float)

    def p_from(nodes):
        s = np.array([strength[n] for n in nodes])
        k = np.array([degree[n] for n in nodes])
        p = np.power(1.0 - w / s, k - 1)
        return np.where(k == 1, 0.0, p)  # a degree-1 node's only edge is always kept

    p = np.minimum(p_from(u), p_from(v))
    nx.set_edge_attributes(H, {(a, b): float(x) for a, b, x in zip(u, v, p)}, "p")
    return H


def disparity_pvalues_dense(S, ids, keep_below=0.1):
    """The same filter on a dense symmetric weight matrix S (zero diagonal), for graphs
    too dense for networkx: the full lyrics similarity graph has 4.6M edges. Returns
    a networkx graph of all nodes and only the edges with p < keep_below (the largest
    alpha we will use), each with its weight and p. For G_lyrics this matters: run on
    a top-k neighbour graph, the filter finds almost nothing, because a song's k
    nearest neighbours all have similar weights; the weak edges it needs as contrast
    are exactly the ones a top-k cut removes."""
    S = np.asarray(S, dtype=float)
    s = S.sum(axis=1)
    k = (S > 0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.power(1.0 - S / s[:, None], (k - 1)[:, None])
    P = np.where(k[:, None] == 1, 0.0, P)   # a degree-1 node's only edge is kept
    P = np.minimum(P, P.T)                  # significant from either endpoint
    i, j = np.nonzero(np.triu((S > 0) & (P < keep_below), 1))
    H = nx.Graph()
    H.add_nodes_from(ids)
    H.add_edges_from((ids[a], ids[b], {"weight": float(S[a, b]), "p": float(P[a, b])})
                     for a, b in zip(i, j))
    return H


def backbone(G, alpha=0.05):
    """Edges with p < alpha (G must come from disparity_pvalues). All nodes are kept,
    so songs that lose every edge show up as isolated nodes."""
    B = nx.Graph()
    B.add_nodes_from(G.nodes(data=True))
    B.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d["p"] < alpha)
    return B


def canonical(G):
    """Copy of G with nodes and edges inserted in sorted order. Louvain and the edge
    swaps visit nodes in insertion order, and that order otherwise depends on Python's
    per-process string hashing (the projections iterate over sets), so the same SEED
    gave different partitions in different runs. Sorting makes runs reproducible."""
    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True), key=lambda nd: str(nd[0])))
    edges = ((u, v, d) if str(u) <= str(v) else (v, u, d) for u, v, d in G.edges(data=True))
    H.add_edges_from(sorted(edges, key=lambda e: (str(e[0]), str(e[1]))))
    return H


def largest_component(G):
    """The largest connected component, as an independent graph in canonical order."""
    return canonical(G.subgraph(max(nx.connected_components(G), key=len)))


# Toy check: a centre with one strong edge (w = 10) and ten weak ones (w = 1); every
# neighbour has a second edge so it is not degree 1, and one node hangs off as a
# pendant. From the centre (k = 11, s = 20) the strong edge has p = 0.5^10 ~ 0.001 and
# is kept; a weak edge has p = 0.95^10 ~ 0.60 from the centre and 0.5 from its other
# end, so it is dropped. The pendant's single edge is kept.
_G = nx.Graph()
_G.add_edge("c", "strong", weight=10.0)
_G.add_edge("strong", "x", weight=1.0)  # so "strong" is not degree 1 (p from it: 1/11)
for _i in range(10):
    _G.add_edge("c", f"weak{_i}", weight=1.0)
    _G.add_edge(f"weak{_i}", f"other{_i}", weight=1.0)
_G.add_edge("other0", "pendant", weight=0.1)
_P = disparity_pvalues(_G)
assert np.isclose(_P["c"]["strong"]["p"], 0.5 ** 10)
assert np.isclose(_P["c"]["weak0"]["p"], min(0.95 ** 10, 0.5))  # weak0: k=2, s=2 -> 0.5
assert _P["other0"]["pendant"]["p"] == 0.0  # degree-1 node keeps its edge
_B = backbone(_P, 0.05)
assert _B.has_edge("c", "strong") and _B.has_edge("other0", "pendant")
assert not _B.has_edge("c", "weak3") and _B.number_of_nodes() == _G.number_of_nodes()


# the dense version gives the same p-values as the graph version
_ids = np.array(sorted(_G.nodes))
_S = nx.to_numpy_array(_G, nodelist=list(_ids))
_D = disparity_pvalues_dense(_S, _ids, keep_below=1.01)
assert all(np.isclose(_D[u][v]["p"], _P[u][v]["p"]) for u, v in _G.edges)
