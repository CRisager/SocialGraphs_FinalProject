"""
The three song graphs (Step 1). Songs are nodes; genre is NEVER used to build an edge.

G_lyrics       TF-IDF vectors of the lyrics, edge weight = cosine similarity between
               every pair of songs. Almost every pair shares some word, so this graph
               is 89% complete (4.6M edges): too big for networkx, but small as a
               dense matrix (3,227 x 3,227), so it is kept as the matrix S and goes
               into networkx only after the disparity filter (backbone.py). We first
               cut it to each song's 20 most similar songs, but the disparity filter
               then finds almost nothing to keep (see disparity_pvalues_dense).
G_instruments  bipartite song-instrument family graph, projected onto the songs with
               Newman's hyperbolic weighting: a family shared by k songs adds
               1 / (k - 1) to each of their pairs, so a family on almost every song
               (guitar, drums) adds almost nothing and a rare shared one adds a lot.
G_combined     songs linked through three feature types: instrument families
               ("inst:"), release country ("country:") and the song's TOP_WORDS
               highest-TF-IDF words ("word:") as lyrical themes. Each type is projected
               on its own (hyperbolic), scaled to the same total edge weight, and the
               three are added (balanced_projection). A single projection of all
               features together let the words take 94% of the backbone's weight: a
               word shared by just two songs gives that pair weight 1, while "guitar"
               (on ~600 songs) gives 0.002, so the graph became a lyrics graph.
"""

from collections import Counter

import networkx as nx
import numpy as np

from .text import lyrics_vectorizer

TOP_WORDS = 10
TFIDF_PARAMS = dict(min_df=2, max_df=0.5)  # sublinear_tf=True is set in lyrics_vectorizer
METADATA = ["song", "artist", "year", "country", "genre_eval", "genre"]


# ---------------------------------------------------------------------------------------
# lyrics
# ---------------------------------------------------------------------------------------

def tfidf(tokens):
    """(X, terms): L2-normalised TF-IDF rows for a Series of token lists. Normalised
    rows make the dot product X @ X.T the cosine similarity."""
    vec = lyrics_vectorizer(**TFIDF_PARAMS)
    X = vec.fit_transform(tokens)
    return X, vec.get_feature_names_out()


def lyrics_similarity(X):
    """Dense cosine-similarity matrix of the songs, with a zero diagonal (no
    self-loops). A zero entry means no shared word, i.e. no edge."""
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    return S


def lyrics_graph(S, ids):
    """networkx graph of every non-zero pair in S. Only for small subsets: the full
    lyrics graph is kept as the matrix."""
    G = nx.Graph()
    G.add_nodes_from(ids)
    i, j = np.nonzero(np.triu(S, 1))
    G.add_weighted_edges_from((ids[a], ids[b], float(S[a, b])) for a, b in zip(i, j))
    return G


def dense_summary(S):
    """summary() for a graph held as a dense weight matrix."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(S)
    n_comp, labels = connected_components(csr_matrix(S > 0), directed=False)
    lcc = np.bincount(labels).max()
    edges = int(np.count_nonzero(np.triu(S, 1)))
    return {"nodes": n, "edges": edges, "density": edges / (n * (n - 1) / 2),
            "components": int(n_comp), "isolated": int(((S > 0).sum(axis=1) == 0).sum()),
            "lcc_nodes": int(lcc), "lcc_share": lcc / n}


def upper_weights(S):
    """The non-zero weights of a dense matrix, one per edge."""
    w = S[np.triu_indices(len(S), 1)]
    return w[w > 0]


def top_words(X, terms, n=TOP_WORDS):
    """Each row's n highest-TF-IDF terms (fewer if the song has fewer distinct words)."""
    X = X.tocsr()
    out = []
    for r in range(X.shape[0]):
        row = X.getrow(r)
        best = row.indices[np.argsort(-row.data, kind="stable")[:n]]
        out.append([terms[t] for t in best])
    return out


# ---------------------------------------------------------------------------------------
# bipartite graphs and their projection
# ---------------------------------------------------------------------------------------

def bipartite_graph(features):
    """Bipartite graph from {song: [feature names]}: songs have bipartite=0, features
    bipartite=1. Feature names must be prefixed ("inst:", ...) so they cannot clash
    with song ids or with each other."""
    B = nx.Graph()
    B.add_nodes_from(features, bipartite=0)
    for song, feats in features.items():
        for f in feats:
            B.add_node(f, bipartite=1)
            B.add_edge(song, f)
    return B


def project(B):
    """Hyperbolic (Newman) projection onto the songs: weight(u, v) = sum over shared
    features f of 1 / (deg(f) - 1). Songs sharing nothing stay as isolated nodes."""
    songs = [n for n, d in B.nodes(data="bipartite") if d == 0]
    return nx.bipartite.collaboration_weighted_projected_graph(B, songs)


def feature_weight_shares(B):
    """Share of the projected graph's total edge weight contributed by each feature
    type (the prefix before ':'). A feature of degree k gives k(k-1)/2 pairs of weight
    1/(k-1) each, i.e. k/2 in total, so this needs no projection."""
    total = Counter()
    for n, d in B.nodes(data="bipartite"):
        k = B.degree(n)
        if d == 1 and k >= 2:
            total[n.split(":", 1)[0]] += k / 2
    s = sum(total.values())
    return {t: v / s for t, v in total.most_common()}


def edge_feature_shares(B, G):
    """Share of the edge weight of G (a projection of B, or a backbone of one) that
    comes from each feature type: each edge's weight is split over its shared
    features, 1 / (deg(f) - 1) each."""
    total = Counter()
    for u, v in G.edges:
        for f in set(B[u]) & set(B[v]):
            total[f.split(":", 1)[0]] += 1 / (B.degree(f) - 1)
    s = sum(total.values())
    return {t: x / s for t, x in total.most_common()}


FEATURE_TYPES = ("inst", "country", "word")


def balanced_projection(features, types=FEATURE_TYPES):
    """Song graph from {song: ['type:name', ...]} in which every feature type has the
    same total edge weight. Each type gets its own bipartite graph and hyperbolic
    projection; the projection's weights are divided by their sum, so each type
    contributes 1 in total, and the types are added edge by edge. Each edge also keeps
    its per-type part as 'w_<type>' (for type_shares). Songs keep their node even if
    no type links them to anyone."""
    G = nx.Graph()
    G.add_nodes_from(features)
    for t in types:
        Bt = bipartite_graph({s: [f for f in fs if f.split(":", 1)[0] == t]
                              for s, fs in features.items()})
        Pt = project(Bt)
        total = Pt.size(weight="weight")
        if total == 0:
            continue
        for u, v, w in Pt.edges(data="weight"):
            d = G.get_edge_data(u, v) or {}
            G.add_edge(u, v, **{**d, "weight": d.get("weight", 0.0) + w / total,
                                f"w_{t}": w / total})
    return G


def type_shares(G, types=FEATURE_TYPES):
    """Share of G's edge weight from each feature type (G from balanced_projection, or
    a backbone of it)."""
    total = {t: sum(d.get(f"w_{t}", 0.0) for *_, d in G.edges(data=True)) for t in types}
    s = sum(total.values())
    return {t: v / s for t, v in total.items()}


def instrument_features(songs):
    """{song: ['inst:<family>', ...]} for the songs with instrument data."""
    fam = songs.instrument_families.dropna()
    return {sid: [f"inst:{f}" for f in v.split(";")] for sid, v in fam.items()}


def combined_features(songs, words):
    """{song: [inst:..., country:..., word:...]} for songs with lyrics, instruments and
    a country. `words` maps song id -> its top TF-IDF words."""
    feats = {}
    for sid, inst in instrument_features(songs).items():
        country = songs.at[sid, "country"]
        if sid in words and isinstance(country, str):
            feats[sid] = inst + [f"country:{country}"] + [f"word:{w}" for w in words[sid]]
    return feats


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def add_metadata(G, songs, cols=METADATA):
    """Copy song metadata onto the nodes, for evaluation and plotting only."""
    for c in cols:
        nx.set_node_attributes(G, songs.loc[list(G.nodes), c].to_dict(), c)
    return G


def summary(G):
    """Size and connectivity of a graph."""
    comps = list(nx.connected_components(G))
    lcc = max(map(len, comps)) if comps else 0
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "density": nx.density(G), "components": len(comps),
            "isolated": nx.number_of_isolates(G), "lcc_nodes": lcc,
            "lcc_share": lcc / max(G.number_of_nodes(), 1)}


# toy checks: songs a, b share a rare feature, all three share a common one
_B = bipartite_graph({"a": ["x:rare", "x:common"], "b": ["x:rare", "x:common"],
                      "c": ["x:common", "y:only"]})
_P = project(_B)
assert set(_P.nodes) == {"a", "b", "c"}
assert np.isclose(_P["a"]["b"]["weight"], 1 + 1 / 2) and np.isclose(_P["a"]["c"]["weight"], 1 / 2)
assert feature_weight_shares(_B) == {"x": 1.0}  # y:only has degree 1, links nobody

_S = np.array([[0, .5, 0], [.5, 0, 0], [0, 0, 0]])
assert dense_summary(_S) == {"nodes": 3, "edges": 1, "density": 1 / 3, "components": 2,
                             "isolated": 1, "lcc_nodes": 2, "lcc_share": 2 / 3}
# on the full projection both share functions agree
_B2 = bipartite_graph({"a": ["x:1", "y:1"], "b": ["x:1", "y:1"], "c": ["x:1"]})
assert all(np.isclose(edge_feature_shares(_B2, project(_B2))[t], v)
           for t, v in feature_weight_shares(_B2).items())

# balanced projection: each type contributes the same total weight, per-type parts add up
_F = {"a": ["inst:x", "word:r", "country:US"], "b": ["inst:x", "word:r", "country:US"],
      "c": ["inst:x", "country:US"], "d": ["country:GB"]}
_G3 = balanced_projection(_F)
assert all(np.isclose(v, 1 / 3) for v in type_shares(_G3).values())
assert all(np.isclose(d["weight"], sum(d.get(f"w_{t}", 0) for t in FEATURE_TYPES))
           for *_, d in _G3.edges(data=True))
assert np.isclose(_G3["a"]["b"]["w_word"], 1.0) and "d" in _G3 and _G3.degree("d") == 0


# ---------------------------------------------------------------------------------------
# descriptive statistics (Step 3)
# ---------------------------------------------------------------------------------------

def degree_stats(G):
    """Degree and clustering of a graph, with two random baselines for the clustering:
    C_rand  = density, what an Erdos-Renyi graph with the same size and density gives;
    C_conf  = (<k^2> - <k>)^2 / (<k>^3 N), the configuration model's expectation for
              the same degree sequence (Newman 2010, eq. 13.47).
    A clustering far above both means triangles beyond what the degrees alone explain."""
    k = np.array([d for _, d in G.degree()], dtype=float)
    n = len(k)
    k1, k2 = k.mean(), (k ** 2).mean()
    comps = list(nx.connected_components(G))
    lcc = max(map(len, comps))
    return {"nodes": n, "edges": G.number_of_edges(), "avg_degree": k1,
            "median_degree": float(np.median(k)), "max_degree": int(k.max()),
            "avg_clustering": nx.average_clustering(G),
            "avg_clustering_weighted": nx.average_clustering(G, weight="weight"),
            "C_rand": nx.density(G), "C_conf": (k2 - k1) ** 2 / (k1 ** 3 * n),
            "components": len(comps), "lcc_nodes": lcc, "lcc_share": lcc / n}


_K = nx.complete_graph(4)
nx.set_edge_attributes(_K, 1.0, "weight")
_d = degree_stats(_K)
assert _d["avg_clustering"] == 1.0 and _d["C_rand"] == 1.0 and _d["avg_degree"] == 3
