"""
Null models (Step 6A, 6B).

6A Configuration-model null for modularity. Even a random graph has partitions with
   M > 0, and Louvain finds them, so Louvain's M on the real graph only means something
   compared with M on random graphs that keep what is not structure: every song's
   degree. Each null graph is the real backbone rewired with double edge swaps
   (nx.double_edge_swap, nswap = SWAPS_PER_EDGE * |E|): pick edges a-b and c-d and
   rewire them to a-d and c-b, which keeps all degrees. The original weights are then
   shuffled randomly over the new edges, so the weight distribution is kept too but no
   longer sits on particular pairs. This is a simplification for weighted graphs: node
   strengths are only kept on average, not exactly.
   Reported: mean and sd of the null M, z = (M_real - mean) / sd, and the empirical
   p-value (1 + #{M_null >= M_real}) / (1 + n), which cannot be 0 with finite n.

6B Permutation test for label agreement. The labels are shuffled over the songs,
   keeping how many songs each label has, and the AMI with the (fixed) communities is
   recomputed each time; the p-value is the share of shuffles at least as high as the
   real AMI, computed the same way as above.
"""

import time

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from .backbone import canonical
from .communities import louvain

SWAPS_PER_EDGE = 10


def rewired(G, seed, swaps_per_edge=SWAPS_PER_EDGE, weight="weight"):
    """Degree-preserving randomisation of G with its weights shuffled over the edges."""
    R = canonical(G)
    nx.double_edge_swap(R, nswap=swaps_per_edge * R.number_of_edges(),
                        max_tries=100 * swaps_per_edge * R.number_of_edges(), seed=seed)
    R = canonical(R)  # fixed edge order, so the weight shuffle is reproducible
    w = np.array([d for *_, d in G.edges(data=weight)])
    w = np.random.default_rng(seed).permutation(w)
    nx.set_edge_attributes(R, {e: float(x) for e, x in zip(R.edges, w)}, weight)
    for _, _, d in R.edges(data=True):  # drop the other edge attributes (p, w_<type>)
        for k in [k for k in d if k != weight]:
            del d[k]
    return R


def modularity_null(G, n, seed, verbose=False):
    """Louvain M on n rewired copies of G (null i uses seed + i)."""
    out = []
    for i in range(n):
        t = time.time()
        R = rewired(G, seed + i)
        comms = louvain(R, seed + i)
        out.append(nx.community.modularity(R, comms, weight="weight"))
        if verbose:
            print(f"  null {i + 1}/{n}: M = {out[-1]:.4f} ({time.time() - t:.1f} s)")
    return np.array(out)


def empirical_p(real, null):
    """(1 + #{null >= real}) / (1 + n): one-sided, never exactly 0."""
    null = np.asarray(null)
    return (1 + np.sum(null >= real)) / (1 + len(null))


def null_summary(real, null):
    sd = null.std(ddof=1)
    return {"M_real": real, "M_null_mean": null.mean(), "M_null_sd": sd,
            "z": (real - null.mean()) / sd if sd > 0 else np.inf,
            "p": empirical_p(real, null), "n_null": len(null)}


def permutation_ami(labels, communities, n, seed):
    """(real AMI, AMI of n label shuffles, p-value) for aligned arrays."""
    labels, communities = np.asarray(labels), np.asarray(communities)
    rng = np.random.default_rng(seed)
    real = adjusted_mutual_info_score(labels, communities)
    null = np.array([adjusted_mutual_info_score(rng.permutation(labels), communities)
                     for _ in range(n)])
    return real, null, empirical_p(real, null)


# toy checks: rewiring keeps degrees and the multiset of weights
_G = nx.random_regular_graph(4, 30, seed=0)
nx.set_edge_attributes(_G, {e: float(i) for i, e in enumerate(_G.edges)}, "weight")
_R = rewired(_G, 0)
assert dict(_R.degree()) == dict(_G.degree()) and set(_R.edges) != set(_G.edges)
assert sorted(w for *_, w in _R.edges(data="weight")) == sorted(w for *_, w in _G.edges(data="weight"))
assert empirical_p(1.0, [0.5] * 99) == 0.01
