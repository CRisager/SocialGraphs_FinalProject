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

Usage:
    python -m src.lyrics_network     # writes figures/lyrics_network.{png,html}
"""

import html

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import FIG_DIR, GREY, SEED, TAG_COLORS, YEARS
from .data import load_data

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


def analyse(name, songs, label_col, verbose=True):
    """Build the network for one song set and compare its communities with the genres.

    Returns a dict with
      metrics    pd.Series of the measures described in the module docstring
      summary    genre mix (row %) and typical words per community (>= 10 songs)
      G, community, genre   the graph and per-node labels, for draw()
    """
    if "title" not in songs:
        songs = songs.rename(columns={"song": "title"})
    G, X, words = build(songs, label_col)
    genre = pd.Series(nx.get_node_attributes(G, "genre"))
    comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=SEED),
                   key=len, reverse=True)
    community = pd.Series({n: i for i, c in enumerate(comms) for n in c}).loc[genre.index]

    genre_sets = [set(genre.index[genre == g]) for g in genre.unique()]
    shares = genre.value_counts(normalize=True)
    table = pd.crosstab(community, genre).rename_axis(index="community", columns=None)
    table = table[table.sum().sort_values(ascending=False).index]  # biggest genre first

    metrics = pd.Series({
        "songs": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "mean degree": 2 * G.number_of_edges() / G.number_of_nodes(),
        "components": nx.number_connected_components(G),
        "communities": len(comms),
        "modularity (Louvain)": nx.community.modularity(G, comms, weight="weight"),
        "genre Q (genres as partition)": nx.community.modularity(G, genre_sets, weight="weight"),
        "same-genre edges": np.mean([genre[a] == genre[b] for a, b in G.edges]),
        "same-genre edges at random": (shares ** 2).sum(),
        "NMI": normalized_mutual_info_score(genre, community),
        "ARI": adjusted_rand_score(genre, community),
        "purity": table.max(axis=1).sum() / len(genre),
        "purity of one big blob": shares.iloc[0],
    }, name=name)

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
    summary = summary[summary.songs >= 10]

    if verbose:
        print(f"\n=== {name} ===")
        print(metrics.round(3).to_string())
        print("\nGenre mix per community (rows sum to 100%):")
        print(summary.to_string())
        small = len(comms) - len(summary)
        if small:
            print(f"(+ {small} communities with fewer than 10 songs)")
    return {"metrics": metrics, "summary": summary,
            "G": G, "community": community, "genre": genre}


def draw(G, community, genre, title, path=FIG_DIR / "lyrics_network.png"):
    """Same layout twice: coloured by Louvain community and by genre. Saves the PNG,
    an interactive HTML next to it, and returns the matplotlib figure. Only the largest
    connected component is drawn: a few isolated songs would squash the layout."""
    giant = G.subgraph(max(nx.connected_components(G), key=len))
    if len(giant) < len(G):
        title += f" - largest component, {len(giant)} of {len(G)} songs"
    G = giant
    pos = nx.spring_layout(G, weight="weight", seed=SEED, iterations=60)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    genres = list(genre.value_counts().index)
    color = {g: TAG_COLORS.get(g, GREY) for g in genres}  # fixed per genre, not per rank
    panels = [("Louvain communities", [cmap(community[n] % 10) for n in G], None),
              ("Genres", [color[genre[n]] for n in G], genres)]
    for ax, (subtitle, colors, legend) in zip(axes, panels):
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.03)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=12, linewidths=0)
        if legend:
            ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=color[g], label=g)
                               for g in legend], fontsize=12)
        ax.set_title(subtitle, fontsize=14)
        ax.axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)

    # Interactive version (same layout): coloured by genre, one legend entry per genre
    # (click to hide), hover a song for its title, artist and community.
    edge_x, edge_y = [], []
    for a, b in G.edges:
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]
    traces = [go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="skip",
                         line={"color": "#cccccc", "width": 0.3}, opacity=0.4,
                         showlegend=False)]
    for g in genres:
        nodes = [n for n in G if genre[n] == g]
        traces.append(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers",
            name=g, marker={"size": 5, "color": color[g]},
            text=[f"<b>{html.escape(str(G.nodes[n]['title']))}</b><br>"
                  f"{html.escape(str(G.nodes[n]['artist']))}<br>"
                  f"genre: {g}<br>community: {community[n]}" for n in nodes],
            hovertemplate="%{text}<extra></extra>"))
    interactive = go.Figure(traces)
    interactive.update_layout(title=title, template="plotly_white", height=900,
                              xaxis={"visible": False}, yaxis={"visible": False})
    interactive.write_html(path.with_suffix(".html"), include_plotlyjs="cdn")
    print(f"Saved {path} and {path.with_suffix('.html').name}")
    return fig


def run_all(songs, genius, verbose=True):
    """The three song sets compared in the report. Returns (metrics table, results)."""
    genius = genius[genius.tag != "rap"]  # only ~400 songs; hip hop starts ~1979
    results = {
        "Genius, all tags": analyse(f"All Genius songs from {YEARS}", genius, "tag", verbose),
        "Genius, without pop": analyse("Genius songs, without pop",
                                       genius[genius.tag != "pop"], "tag", verbose),
        "MusicBrainz, labels agree": analyse(
            "MusicBrainz songs where both sources agree on the genre",
            songs[songs.labels_agree == True], "label", verbose),  # noqa: E712
    }
    metrics = pd.DataFrame({name: r["metrics"] for name, r in results.items()})
    return metrics, results


def main():
    songs, genius = load_data()
    _, results = run_all(songs, genius)
    r = results["Genius, without pop"]
    draw(r["G"], r["community"], r["genre"],
         f"Lyrics similarity network, Genius {YEARS} without pop "
         f"(each song linked to its {K} most similar)")


if __name__ == "__main__":
    main()
