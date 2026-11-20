"""
Lyric-similarity network: do songs (and artists) with similar lyrics share a genre?

Nodes are songs, or artists in the artist-level variant. Lyrics become TF-IDF vectors
over word tokens with English stop words removed (dataCleaner/text.py), and every node
is linked to the k nodes whose lyrics are most similar (cosine similarity = edge
weight). In the mutual variant, two nodes are only linked when each is among the
other's k most similar: this removes the hubs (generic "baby, I love you" songs that are
everybody's neighbour, up to 205 links at k = 10) that tie the plain graph into one
hairball, at the cost of leaving some songs without links. Two songs by the same artist are never linked, so a community cannot simply
be one artist's catalogue. An artist's vector is the mean of its songs' vectors, and
its genre is the most common genre among its songs.

Genre is the 5-label MusicBrainz scheme mb_label from combine.py (rock, rb, pop,
country, rap) with rap dropped (19 songs).

For each graph (songs at k = 5, 10, 20, as the k sensitivity check; artists at k = 10;
each plain and mutual):
  same-genre   share of edges joining two nodes of the same genre, against 1,000
               shuffles of the genre labels over the nodes (z-score)
  <genre>_lift share of a genre's links that stay inside the genre, divided by the
               genre's share of the nodes (1 = no lyrical identity, 2 = twice chance)
  genre Q      modularity of the genres used as the partition
  Louvain      run with 10 seeds: modularity, number of communities and NMI against
               genre (mean and sd over seeds); stability = mean NMI between the
               partitions of different seeds
  NMI null     NMI of the best-modularity partition against 1,000 genre shuffles
               (z-score); NMI is 0 for no shared information and 1 for identical
  rewired      the same numbers on 10 degree-preserving rewirings of the graph (double
               edge swaps, 10 per edge; the edge weights are shuffled over the new
               edges). Every node keeps its number of links but loses who they go to,
               so this asks whether the community structure and the same-genre links
               come from lyric content or just from which songs have many links
Results go to clustering/lyrics_network_results.csv (one row per graph); the genre mix
and typical words of each community, and the figure, are for the mutual song graph
at k = 10 (songs without links are left out of the figure); figures/lyrics_genre_lift.png
shows each genre's lift for the k = 10 graphs.

Needs songs_<years>_lyrics.csv (see dataCleaner/).

Balanced check (--balanced, about 10 minutes): rock is 58% of the songs, so every number
above is dominated by it. This repeats the song graphs (k = 10, and mutual k = 20) on 5
random samples with the same number of songs per genre (all of the smallest genre),
so chance is 25% for every genre; results go to lyrics_network_balanced.csv.

Usage:
    python clustering/lyrics_network.py              # everything except the balanced check
    python clustering/lyrics_network.py --balanced   # only the balanced check
"""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from pyvis.network import Network
from scipy.sparse import csr_matrix
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataCleaner"))
from clean import YEARS, artist_main  # noqa: E402  (the year range is set in clean.py)
from text import lyrics_vectorizer  # noqa: E402
FIG_DIR = Path(__file__).resolve().parent / "figures"
RESULTS = Path(__file__).resolve().parent / "lyrics_network_results.csv"
BALANCED = Path(__file__).resolve().parent / "lyrics_network_balanced.csv"
N_DRAWS = 5
LABELS = ["rock", "rb", "pop", "country"]  # mb_label without rap
K_VALUES = [5, 10, 20]
ARTIST_K = 10
SEEDS = range(10)
N_SHUFFLES = 1000
N_REWIRES = 10
SWAPS_PER_EDGE = 10
# shortcut: build() keeps a dense n x n similarity matrix (8k songs = 0.25 GB), so larger
# sets are randomly sampled down; switch to sklearn NearestNeighbors to use them all
MAX_NODES = 8000


def neighbours(sim, k_max):
    """Each row's k_max most similar columns, most similar first."""
    idx = np.argpartition(-sim, k_max, axis=1)[:, :k_max]
    order = np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def knn_graph(sim, nbrs, k, mutual=False):
    """Undirected graph linking every node to its k most similar nodes (the first k
    columns of nbrs); if mutual, only pairs that chose each other. Pairs with
    similarity 0 are never linked."""
    G = nx.Graph()
    G.add_nodes_from(range(len(sim)))
    chosen = {(i, int(j)) for i, row in enumerate(nbrs[:, :k]) for j in row if sim[i, j] > 0}
    G.add_weighted_edges_from((i, j, float(sim[i, j])) for i, j in chosen
                              if not mutual or (j, i) in chosen)
    return G


# 0 and 1 choose each other, 3 chooses 1, 2 is similar to nothing
_sim = np.array([[0, .9, 0, 0], [.9, 0, 0, .1], [0, 0, 0, 0], [0, .1, 0, 0]])
assert sorted(knn_graph(_sim, neighbours(_sim, 1), 1).edges) == [(0, 1), (1, 3)]
assert sorted(knn_graph(_sim, neighbours(_sim, 1), 1, mutual=True).edges) == [(0, 1)]


def rewired(G, seed):
    """G with its edges rewired by double edge swaps (every node keeps its degree) and
    its edge weights shuffled over the new edges."""
    R = G.copy()
    nx.double_edge_swap(R, nswap=SWAPS_PER_EDGE * R.number_of_edges(),
                        max_tries=10 ** 9, seed=seed)
    weights = np.random.default_rng(seed).permutation([w for *_, w in G.edges(data="weight")])
    nx.set_edge_attributes(R, dict(zip(R.edges, weights)), "weight")
    return R


_G = nx.random_regular_graph(4, 30, seed=0)
nx.set_edge_attributes(_G, 1.0, "weight")
_R = rewired(_G, 0)
assert dict(_R.degree()) == dict(_G.degree()) and set(_R.edges) != set(_G.edges)


def labels_of(comms, n):
    """Community number of each node 0..n-1, biggest community first."""
    part = np.empty(n, dtype=int)
    for c, nodes in enumerate(sorted(comms, key=len, reverse=True)):
        part[list(nodes)] = c
    return part


def similarity(X, groups=None):
    """Cosine similarity of the unit-length rows of X, with pairs in the same group
    (and each node with itself) set to 0."""
    sim = (X @ X.T).toarray().astype(np.float32)
    if groups is None:
        np.fill_diagonal(sim, 0)
    else:
        codes = pd.factorize(groups)[0]
        sim[codes[:, None] == codes[None, :]] = 0  # also clears the diagonal
    return sim


def load():
    songs = pd.read_csv(ROOT / f"songs_{YEARS}_lyrics.csv").rename(columns={"song": "title"})
    songs = songs[songs.lyrics.notna() & songs.mb_label.isin(LABELS)]
    if len(songs) > MAX_NODES:
        print(f"(random sample of {MAX_NODES} of {len(songs)} songs)")
        songs = songs.sample(MAX_NODES, random_state=0)
    # "Merle Haggard" and "Merle Haggard and The Strangers" are one artist
    return songs.assign(artist_id=songs.artist.map(artist_main)).reset_index(drop=True)


def shuffle_test(stat, labels, rng):
    """Observed stat(labels), and the mean, sd and z-score over label shuffles."""
    observed = stat(labels)
    null = np.array([stat(rng.permutation(labels)) for _ in range(N_SHUFFLES)])
    return observed, null.mean(), null.std(), (observed - null.mean()) / null.std()


def analyse(name, G, genre, k, mutual):
    """One row of results; also returns the best-modularity Louvain partition."""
    rng = np.random.default_rng(0)
    genre = np.asarray(genre)
    u, v = np.array(G.edges).T
    same = shuffle_test(lambda g: np.mean(g[u] == g[v]), genre, rng)
    genre_sets = [set(np.flatnonzero(genre == g)) for g in LABELS]
    ends_u, ends_v = np.r_[u, v], np.r_[v, u]  # every edge seen from both ends
    lift = {f"{g}_lift": np.mean(genre[ends_v[genre[ends_u] == g]] == g) / np.mean(genre == g)
            for g in LABELS}

    runs = []
    for seed in SEEDS:
        comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
        runs.append((nx.community.modularity(G, comms, weight="weight"),
                     labels_of(comms, len(genre))))
    mods = np.array([m for m, _ in runs])
    nmis = np.array([nmi(genre, p) for _, p in runs])
    best = runs[mods.argmax()][1]
    stability = np.mean([nmi(a, b) for (_, a), (_, b) in combinations(runs, 2)])
    nmi_null = shuffle_test(lambda g: nmi(g, best), genre, rng)

    rew = []
    for seed in range(N_REWIRES):
        R = rewired(G, seed)
        ru, rv = np.array(R.edges).T
        comms = nx.community.louvain_communities(R, weight="weight", seed=seed)
        rew.append([np.mean(genre[ru] == genre[rv]),
                    nx.community.modularity(R, comms, weight="weight"),
                    nmi(genre, labels_of(comms, len(genre)))])
    rew_mean, rew_sd = np.mean(rew, axis=0), np.std(rew, axis=0)

    row = {"graph": name, "mutual": mutual, "k": k, "nodes": G.number_of_nodes(),
           "edges": G.number_of_edges(), "isolated": nx.number_of_isolates(G),
           "max_degree": max(d for _, d in G.degree()),
           "components": nx.number_connected_components(G),
           "same_genre": same[0], "same_genre_null": same[1], "same_genre_z": same[3],
           **lift,
           "rewired_same_genre": rew_mean[0],
           "rewired_same_genre_z": (same[0] - rew_mean[0]) / rew_sd[0],
           "genre_modularity": nx.community.modularity(G, genre_sets, weight="weight"),
           "louvain_modularity": mods.mean(), "louvain_modularity_sd": mods.std(),
           "communities": np.mean([p.max() + 1 for _, p in runs]),
           "nmi": nmis.mean(), "nmi_sd": nmis.std(), "seed_stability_nmi": stability,
           "nmi_best": nmi_null[0], "nmi_null": nmi_null[1], "nmi_null_sd": nmi_null[2],
           "nmi_z": nmi_null[3], "rewired_modularity": rew_mean[1],
           "rewired_modularity_sd": rew_sd[1], "rewired_nmi": rew_mean[2]}
    print(f"\n=== {name}, k = {k}{', mutual' if mutual else ''} ===")
    print(f"{row['nodes']} nodes, {row['edges']} edges, {row['isolated']} without links, "
          f"max degree {row['max_degree']}, {row['components']} component(s)")
    print(f"Same-genre edges {same[0]:.1%} (shuffled genres: {same[1]:.1%} ± {same[2]:.1%}, "
          f"z = {same[3]:.0f}; rewired: {rew_mean[0]:.1%} ± {rew_sd[0]:.1%}, "
          f"z = {row['rewired_same_genre_z']:.0f}); genres as partition: modularity "
          f"{row['genre_modularity']:.3f}")
    print("Links staying in the genre, relative to chance: "
          + ", ".join(f"{g} {lift[g + '_lift']:.2f}" for g in LABELS))
    print(f"Louvain over {len(SEEDS)} seeds: modularity {mods.mean():.3f} ± {mods.std():.3f}, "
          f"{row['communities']:.0f} communities, NMI with genre {nmis.mean():.3f} ± "
          f"{nmis.std():.3f}, NMI between seeds {stability:.2f}")
    print(f"Best partition: NMI {nmi_null[0]:.3f} vs {nmi_null[1]:.4f} ± {nmi_null[2]:.4f} "
          f"for shuffled genres (z = {nmi_null[3]:.0f})")
    print(f"Rewired ({N_REWIRES} graphs): Louvain modularity {rew_mean[1]:.3f} ± {rew_sd[1]:.3f}, "
          f"NMI with genre {rew_mean[2]:.3f}")
    return row, best


def describe(community, genre, X, words):
    """Genre mix and the most typical words (highest mean TF-IDF inside the community
    relative to all songs) of each community."""
    table = pd.crosstab(community, genre).rename_axis(index="community", columns=None)
    table = table[[g for g in LABELS if g in table]]
    overall = np.asarray(X.mean(axis=0)).ravel()
    summary = table.div(table.sum(axis=1), axis=0).mul(100).round().astype(int).astype(str) + "%"
    summary.insert(0, "songs", table.sum(axis=1))
    summary["typical words"] = [
        ", ".join(words[np.argsort(np.asarray(X[community == c].mean(axis=0)).ravel()
                                   - overall)[-8:][::-1]]) for c in table.index]
    print("\nGenre mix per community (rows sum to 100%):")
    print(summary[summary.songs >= 10].to_string())
    if (summary.songs < 10).any():
        print(f"(+ {(summary.songs < 10).sum()} communities with fewer than 10 songs)")


def draw(G, community, genre, songs, title, path):
    """Same layout twice: coloured by Louvain community and by genre."""
    pos = nx.spring_layout(G, weight="weight", seed=0, iterations=60)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    panels = [("Louvain communities", [cmap(community[n] % 10) for n in G], None),
              ("Genres", [cmap(LABELS.index(genre[n])) for n in G], LABELS)]
    for ax, (subtitle, colors, legend) in zip(axes, panels):
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.03)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=12, linewidths=0)
        if legend:
            ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=cmap(i), label=g)
                               for i, g in enumerate(legend)], fontsize=12)
        ax.set_title(subtitle, fontsize=14)
        ax.axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"\nSaved {path}")

    # Interactive version (same layout, physics off so it opens instantly): coloured by
    # genre; hover a song for details, the filter menu can show one community or genre.
    net = Network(height="900px", width="100%", cdn_resources="remote",
                  select_menu=True, filter_menu=True)
    for n in G:
        x, y = pos[n]
        net.add_node(n, label=" ", title=f"{songs.title[n]}\n{songs.artist[n]}\n"
                                         f"genre: {genre[n]}\ncommunity: {community[n]}",
                     genre=genre[n], community=int(community[n]), size=8,
                     color=to_hex(cmap(LABELS.index(genre[n]))), x=3000 * x, y=3000 * y)
    for a, b, w in G.edges(data="weight"):
        net.add_edge(a, b, width=0.3, title=f"similarity: {w:.2f}", color="#e8e8e8")
    net.toggle_physics(False)
    net.set_edge_smooth("false")
    net.write_html(str(path.with_suffix(".html")), notebook=False)
    print(f"Saved {path.with_suffix('.html')}")


def draw_lift(results, path):
    """Per genre: share of its links that stay in the genre relative to chance, for the
    k = 10 song and artist graphs (1 = genre has no lyrical identity)."""
    rows = results[results.k == 10]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(rows)
    for i, (_, r) in enumerate(rows.iterrows()):
        ax.bar(np.arange(len(LABELS)) + i * width, [r[f"{g}_lift"] for g in LABELS], width,
               label=f"{r.graph}{', mutual' if r.mutual else ''}")
    ax.axhline(1, color="black", lw=1, ls="--")
    ax.set_xticks(np.arange(len(LABELS)) + 0.4 - width / 2, LABELS)
    ax.set_yscale("log")
    ax.set_ylabel("same-genre links / chance")
    ax.set_title("Links that stay inside the genre, k = 10")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def main():
    songs = load()
    print(f"{len(songs)} songs with lyrics by {songs.artist_id.nunique()} artists, "
          f"genres: {songs.mb_label.value_counts().to_dict()}")
    vectorizer = lyrics_vectorizer(min_df=2)
    X = vectorizer.fit_transform(songs.lyrics)  # rows are unit length
    words = vectorizer.get_feature_names_out()
    print(f"Vocabulary: {len(words)} words")

    rows = []
    sim = similarity(X, songs.artist_id)
    nbrs = neighbours(sim, max(K_VALUES))
    for mutual in (False, True):
        for k in K_VALUES:
            G = knn_graph(sim, nbrs, k, mutual)
            row, best = analyse("songs", G, songs.mb_label, k, mutual)
            rows.append(row)
            if mutual and k == 10:
                describe(best, songs.mb_label.to_numpy(), X, words)
                draw(G.subgraph(n for n, d in G.degree() if d), best,
                     songs.mb_label.to_numpy(), songs,
                     f"Lyric-similarity network, MusicBrainz songs {YEARS} (songs linked when "
                     f"each is among the other's {k} most similar)",
                     FIG_DIR / "lyrics_network.png")
    del sim

    # artist level: mean song vector per artist, labelled with its most common genre
    codes, artists = pd.factorize(songs.artist_id)
    member = csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))))
    A = normalize(member @ X)  # sum and mean point the same way
    artist_genre = songs.groupby(codes).mb_label.agg(lambda g: g.value_counts().index[0])
    sim = similarity(A)
    nbrs = neighbours(sim, ARTIST_K)
    for mutual in (False, True):
        G = knn_graph(sim, nbrs, ARTIST_K, mutual)
        rows.append(analyse("artists", G, artist_genre, ARTIST_K, mutual)[0])

    results = pd.DataFrame(rows).round(4)
    results.to_csv(RESULTS, index=False)
    print(f"\nSaved {RESULTS}")
    draw_lift(results, FIG_DIR / "lyrics_genre_lift.png")


def balanced_sample(songs, seed):
    """The same number of random songs from every genre: as many as the smallest has."""
    n = songs.mb_label.value_counts().min()
    return songs.groupby("mb_label").sample(n, random_state=seed).reset_index(drop=True)


def main_balanced():
    songs = load()
    rows = []
    for draw_no in range(N_DRAWS):
        sample = balanced_sample(songs, draw_no)
        print(f"\n##### draw {draw_no + 1} of {N_DRAWS}: {len(sample)} songs, "
              f"{sample.mb_label.value_counts().to_dict()}")
        sim = similarity(lyrics_vectorizer(min_df=2).fit_transform(sample.lyrics),
                         sample.artist_id)
        nbrs = neighbours(sim, 20)
        for k, mutual in ((10, False), (20, True)):
            row = analyse("songs, balanced", knn_graph(sim, nbrs, k, mutual),
                          sample.mb_label, k, mutual)[0]
            rows.append({"draw": draw_no, **row})
    results = pd.DataFrame(rows).round(4)
    results.to_csv(BALANCED, index=False)
    cols = ["same_genre", "same_genre_null", "rewired_same_genre", *[f"{g}_lift" for g in LABELS],
            "louvain_modularity", "rewired_modularity", "nmi", "nmi_null", "rewired_nmi"]
    summary = results.groupby(["k", "mutual"])[cols].agg(["mean", "std"]).round(3).T
    print(f"\nMean and sd over {N_DRAWS} balanced draws:\n{summary.to_string()}")
    print(f"Saved {BALANCED}")


assert (balanced_sample(pd.DataFrame({"mb_label": list("aaab"), "x": range(4)}), 0)
        .mb_label.value_counts().to_dict() == {"a": 1, "b": 1})


if __name__ == "__main__":
    main_balanced() if "--balanced" in sys.argv else main()
