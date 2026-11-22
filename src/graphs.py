"""
The three song graphs (Step 1). Songs are nodes; genre is NEVER used to build an edge.

G_lyrics       TF-IDF vectors of the lyrics, edge weight = cosine similarity. The full
               similarity matrix of the 3,234 songs with lyrics is 91% non-zero (4.8M
               edges, since almost every pair shares some word), too big for networkx
               and the null models, so every song keeps only its K_NEIGHBOURS most
               similar songs as candidate edges (an edge is kept if either song chose
               it). The disparity filter (Step 2) then decides which edges matter.
G_instruments  bipartite song-instrument family graph, projected onto the songs with
               Newman's hyperbolic weighting: a family shared by k songs adds
               1 / (k - 1) to each of their pairs, so a family on almost every song
               (guitar, drums) adds almost nothing and a rare shared one adds a lot.
G_combined     the same projection of a bipartite graph whose feature nodes are the
               song's instrument families ("inst:"), its release country ("country:")
               and its TOP_WORDS highest-TF-IDF words ("word:") as lyrical themes.
"""

from collections import Counter

import networkx as nx
import numpy as np

from .text import lyrics_vectorizer

K_NEIGHBOURS = 20
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


def lyrics_graph(X, ids, k=K_NEIGHBOURS):
    """Song graph weighted by cosine similarity. With k, each song keeps its k most
    similar songs (union over both endpoints); k=None keeps every non-zero pair.
    Self-loops and zero similarities are never edges."""
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    G = nx.Graph()
    G.add_nodes_from(ids)
    if k is None:
        i, j = np.nonzero(np.triu(S, 1))
    else:
        k = min(k, len(ids) - 1)
        nbrs = np.argpartition(-S, k - 1, axis=1)[:, :k]  # k largest per row, unordered
        i = np.repeat(np.arange(len(ids)), k)
        j = nbrs.ravel()
        keep = S[i, j] > 0
        i, j = i[keep], j[keep]
    G.add_weighted_edges_from((ids[a], ids[b], float(S[a, b])) for a, b in zip(i, j))
    return G


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
