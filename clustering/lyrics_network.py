"""
Lyrics similarity network: do songs with similar lyrics end up in the same genre?

Each song is a node. Lyrics are turned into TF-IDF vectors, and every song is linked
to the K songs whose lyrics are most similar to its own (cosine similarity = edge
weight). Louvain then finds communities without seeing the genres, and we measure
how well the communities match the genre labels:

  modularity   how clearly the network splits into the Louvain communities
  genre Q      modularity of the genre labels used as the partition (0 = genres are
               spread over the network at random)
  same-genre   share of edges joining two songs of the same genre, next to the share
               expected if the genres were placed at random
  NMI / ARI    agreement between communities and genres (0 = none, 1 = identical)
  purity       share of songs that have their community's most common genre, next to
               the share of the most common genre overall (what one big blob scores)

Two songs by the same artist are never linked, so a community cannot simply be one
artist's catalogue.

Needs songs_<years>_lyrics.csv and genius_<years>_clean.csv (see dataCleaner/).

Usage:
    python clustering/lyrics_network.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataCleaner"))
from clean import YEARS  # noqa: E402  (the year range is set in clean.py)
FIG_DIR = Path(__file__).resolve().parent / "figures"
SEED = 42
K = 10  # each song is linked to its K most similar songs
# shortcut: build() keeps a dense n x n similarity matrix (8k songs = 0.5 GB), so larger
# sets are randomly sampled down; switch to sklearn NearestNeighbors to use them all
MAX_SONGS = 8000


def knn_graph(sim, k):
    """Undirected graph linking every node to its k highest-similarity nodes.
    Pairs with similarity 0 are never linked."""
    G = nx.Graph()
    G.add_nodes_from(range(len(sim)))
    for i, row in enumerate(sim):
        for j in np.argsort(row)[-k:]:
            if row[j] > 0:
                G.add_edge(i, int(j), weight=float(row[j]))
    return G


# 0 and 1 are similar, 2 is similar to nothing: one edge, node 2 isolated
assert list(knn_graph(np.array([[0, .9, 0], [.9, 0, 0], [0, 0, 0]]), 1).edges) == [(0, 1)]


def build(songs, label_col):
    """Returns the graph (node attributes: title, artist, genre) and what is needed
    to describe communities by their words."""
    songs = songs.dropna(subset=["lyrics", label_col])
    if len(songs) > MAX_SONGS:
        print(f"(random sample of {MAX_SONGS} of {len(songs)} songs)")
        songs = songs.sample(MAX_SONGS, random_state=SEED)
    songs = songs.reset_index(drop=True)
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(songs.lyrics)  # rows are unit length
    sim = (X @ X.T).toarray()
    artist = songs.artist.to_numpy()
    sim[artist[:, None] == artist[None, :]] = 0  # also clears the diagonal
    G = knn_graph(sim, K)
    for i, row in songs.iterrows():
        G.nodes[i].update(title=row.title, artist=row.artist, genre=row[label_col])
    return G, X, vectorizer.get_feature_names_out()


def analyse(name, songs, label_col):
    G, X, words = build(songs, label_col)
    genre = pd.Series(nx.get_node_attributes(G, "genre"))
    comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=SEED),
                   key=len, reverse=True)
    community = pd.Series({n: i for i, c in enumerate(comms) for n in c}).loc[genre.index]

    genre_sets = [set(genre.index[genre == g]) for g in genre.unique()]
    same = np.mean([genre[a] == genre[b] for a, b in G.edges])
    table = pd.crosstab(community, genre).rename_axis(index="community", columns=None)
    table = table[table.sum().sort_values(ascending=False).index]  # biggest genre first

    print(f"\n=== {name} ===")
    print(f"{G.number_of_nodes()} songs, {G.number_of_edges()} edges, "
          f"mean degree {2 * G.number_of_edges() / G.number_of_nodes():.1f}, "
          f"{nx.number_connected_components(G)} component(s)")
    print(f"Louvain: {len(comms)} communities, modularity "
          f"{nx.community.modularity(G, comms, weight='weight'):.3f}")
    print(f"Genres as partition: modularity "
          f"{nx.community.modularity(G, genre_sets, weight='weight'):.3f}")
    print(f"Same-genre edges: {same:.1%} (expected at random: "
          f"{(genre.value_counts(normalize=True) ** 2).sum():.1%})")
    print(f"Communities vs genres: NMI {normalized_mutual_info_score(genre, community):.3f}, "
          f"ARI {adjusted_rand_score(genre, community):.3f}, "
          f"purity {table.max(axis=1).sum() / len(genre):.1%} "
          f"(largest genre alone: {genre.value_counts(normalize=True).iloc[0]:.1%})")

    # what each community is "about": words with the highest mean TF-IDF inside it
    # relative to the whole collection
    overall = np.asarray(X.mean(axis=0)).ravel()
    top_words = {}
    for i, comm in enumerate(comms):
        inside = np.asarray(X[sorted(comm)].mean(axis=0)).ravel()
        top_words[i] = ", ".join(words[np.argsort(inside - overall)[-8:][::-1]])
    summary = table.div(table.sum(axis=1), axis=0).mul(100).round().astype(int).astype(str) + "%"
    summary.insert(0, "songs", table.sum(axis=1))
    summary["typical words"] = pd.Series(top_words)
    print("\nGenre mix per community (rows sum to 100%):")
    print(summary[summary.songs >= 10].to_string())
    small = (summary.songs < 10).sum()
    if small:
        print(f"(+ {small} communities with fewer than 10 songs)")
    return G, community, genre


def draw(G, community, genre, title, path):
    """Same layout twice: coloured by Louvain community and by genre."""
    pos = nx.spring_layout(G, weight="weight", seed=SEED, iterations=60)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    genres = list(genre.value_counts().index)
    panels = [("Louvain communities", [cmap(community[n] % 10) for n in G], None),
              ("Genres", [cmap(genres.index(genre[n])) for n in G], genres)]
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
        d = G.nodes[n]
        net.add_node(n, label=" ", title=f"{d['title']}\n{d['artist']}\ngenre: {genre[n]}\n"
                                         f"community: {community[n]}",
                     genre=genre[n], community=int(community[n]), size=8,
                     color=to_hex(cmap(genres.index(genre[n]))), x=3000 * x, y=3000 * y)
    for a, b, w in G.edges(data="weight"):
        net.add_edge(a, b, width=0.3, title=f"similarity: {w:.2f}", color="#e8e8e8")
    net.toggle_physics(False)
    net.set_edge_smooth("false")
    net.write_html(str(path.with_suffix(".html")), notebook=False)
    print(f"Saved {path.with_suffix('.html')}")


def main():
    matched = pd.read_csv(ROOT / f"songs_{YEARS}_lyrics.csv").rename(columns={"song": "title"})
    genius = pd.read_csv(ROOT / f"genius_{YEARS}_clean.csv")
    genius = genius[genius.tag != "rap"]  # only 18 songs in 1970; hip hop starts ~1979

    analyse(f"All Genius songs from {YEARS}", genius, "tag")
    result = analyse("Genius songs, without pop", genius[genius.tag != "pop"], "tag")
    draw(*result, f"Lyrics similarity network, Genius {YEARS} without pop "
                  f"(each song linked to its {K} most similar)",
         FIG_DIR / "lyrics_network.png")
    analyse("MusicBrainz songs where both sources agree on the genre",
            matched[matched.labels_agree == True], "label")  # noqa: E712


if __name__ == "__main__":
    main()
