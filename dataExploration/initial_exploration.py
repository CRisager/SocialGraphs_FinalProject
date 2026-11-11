# %% [markdown]
# # Initial exploration of songs_1970.csv
# Table overview (NaNs, distributions) followed by two networks built from the
# MusicBrainz genre tags, with Louvain communities:
# - artist network: artists linked by genre overlap (Jaccard similarity)
# - genre network: genres linked when the same artist is tagged with both

# %%
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.colors import to_hex
from pyvis.network import Network

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running in a notebook / interactive window
    HERE = Path.cwd()
DATA = next(p / "songs_1970.csv" for p in [HERE, *HERE.parents] if (p / "songs_1970.csv").exists())
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
JACCARD_THRESHOLD = 0.3  # min genre overlap for an artist-artist edge

df = pd.read_csv(DATA)
df.head()

# %% [markdown]
# ## Table overview

# %%
print(f"Shape: {df.shape}")
print(df.dtypes, "\n")

overview = pd.DataFrame({
    "nan": df.isna().sum(),
    "nan_%": (df.isna().mean() * 100).round(1),
    "unique": df.nunique(),
})
print(overview, "\n")

# year and language are constant by construction of the crawler
for col in ["year", "language"]:
    print(f"{col}: {df[col].unique()}")

print(f"\nDuplicate ids: {df.id.duplicated().sum()}")
# same recording released in several countries gets a separate MusicBrainz id
dupes = df[df.duplicated(["song", "artist"], keep=False)]
print(f"Duplicate (song, artist) pairs: {df.duplicated(['song', 'artist']).sum()} "
      f"(spread over {dupes.country.nunique()} countries, e.g. "
      f"{dupes.iloc[0].song!r} by {dupes.iloc[0].artist})")
print(f"Song titles used by more than one artist: "
      f"{(df.groupby('song').artist.nunique() > 1).sum()}")
print(f"Artist credits that look like collaborations (&, and, feat.): "
      f"{df.artist.drop_duplicates().str.contains(r'&| and |feat', case=False).sum()}")
print(f"Words per title: {df.song.str.split().str.len().describe().round(1).to_dict()}")

# %% [markdown]
# ### Where are the missing genres?
# Genres come from the artist's MusicBrainz tags, so a NaN means the artist has
# no (known) genre tags. Check whether it is concentrated in some countries.

# %%
df["has_genre"] = df.genres.notna()
artists = df.groupby("artist").agg(
    songs=("id", "size"),
    country=("country", lambda s: s.mode().iat[0]),
    genres=("genres", "first"),  # first non-NaN; a few artists are tagged on only some songs
)
artists["has_genre"] = artists.genres.notna()
print(f"Artists: {len(artists)}, with genres: {artists.has_genre.sum()} "
      f"({artists.has_genre.mean():.0%})")

by_country = df.groupby("country").has_genre.agg(songs="size", genre_coverage="mean")
print(by_country.sort_values("songs", ascending=False).head(15).round(2))

# artists with many songs are more likely to be tagged
print(artists.groupby(pd.cut(artists.songs, [0, 1, 2, 5, 100])).has_genre.mean().round(2))

# %% [markdown]
# ### How many songs have lyrics in the Genius dataset?
# Naive match on normalised title + artist (lowercase, alphanumerics only,
# parentheses dropped) against genius_songs_trimmed.csv (Genius year == 1970).

# %%
def normalise(s):
    return (s.str.lower().str.replace("’", "'")
            .str.replace(r"\(.*?\)", "", regex=True)
            .str.replace(r"[^a-z0-9]", "", regex=True))


genius = pd.read_csv(DATA.parent / "genius_songs_trimmed.csv", usecols=["title", "artist"])
genius_keys = set(normalise(genius.title) + "|" + normalise(genius.artist))
df["has_lyrics"] = (normalise(df.song) + "|" + normalise(df.artist)).isin(genius_keys)

print(f"Genius songs from 1970: {len(genius)}")
print(f"Matched songs: {df.has_lyrics.sum()} ({df.has_lyrics.mean():.0%})")
print(f"Artists also on Genius (1970): "
      f"{normalise(df.artist).isin(set(normalise(genius.artist))).mean():.0%}")
print("Match rate by genre tags present:", df.groupby("has_genre").has_lyrics.mean().round(2).to_dict())

# %% [markdown]
# ### Distributions

# %%
genre_lists = df.genres.dropna().str.split(";")
genre_counts = genre_lists.explode().value_counts()
print(f"Distinct genres: {len(genre_counts)}")
print(f"Genres per song: {genre_lists.str.len().describe().round(2).to_dict()}")

BLUE, GREY = "#2a78d6", "#c9c8c3"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.axisbelow": True})
top_countries = df.country.value_counts().index[:15]

fig, axes = plt.subplots(3, 3, figsize=(17, 14))

songs_per_artist = artists.songs.value_counts().sort_index()
axes[0, 0].loglog(songs_per_artist.index, songs_per_artist.values, "o", color=BLUE)
axes[0, 0].set(title="Songs per artist", xlabel="songs", ylabel="artists")

# missing genres per country, stacked so both volume and coverage are visible
tagged = pd.DataFrame({
    "genre tags": df[df.has_genre].country.value_counts(),
    "no genre tags (NaN)": df[~df.has_genre].country.value_counts(),
}).reindex(top_countries).fillna(0)
tagged.plot.bar(ax=axes[0, 1], stacked=True, color=[BLUE, GREY], width=0.8,
                title="Songs per country (top 15)", xlabel="")

artists.songs.nlargest(15).iloc[::-1].plot.barh(
    ax=axes[0, 2], color=BLUE, title="Artists with most songs", ylabel="")

genre_counts.head(15).iloc[::-1].plot.barh(
    ax=axes[1, 0], color=BLUE, title="Most common genre tags (songs)", ylabel="")

df.genre.value_counts().head(15).iloc[::-1].plot.barh(
    ax=axes[1, 1], color=BLUE, title="Primary genre (`genre` column)", ylabel="")

axes[1, 2].loglog(range(1, len(genre_counts) + 1), genre_counts.values, color=BLUE, lw=2)
axes[1, 2].set(title=f"Genre tag rank vs. frequency ({len(genre_counts)} tags)",
               xlabel="rank", ylabel="songs")

axes[2, 0].hist(genre_lists.str.len(), bins=range(1, 27), color=BLUE, rwidth=0.85)
axes[2, 0].set(title="Genre tags per song", xlabel="tags", ylabel="songs")

lyrics_rate = df.groupby("country").has_lyrics.mean().loc[top_countries[:10]]
lyrics_rate.plot.bar(ax=axes[2, 1], color=BLUE, width=0.7, xlabel="", ylim=(0, 1),
                     title=f"Songs matched to Genius lyrics (overall {df.has_lyrics.mean():.0%})")
axes[2, 1].axhline(df.has_lyrics.mean(), color="#52514e", lw=1, ls="--")
axes[2, 1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))

# primary genre mix per country, rows normalised to shares
mix = pd.crosstab(df.country, df.genre, normalize="index")
mix = mix.loc[top_countries[:8], df.genre.value_counts().index[:8]]
im = axes[2, 2].imshow(mix.values, cmap="Blues", vmin=0, vmax=mix.values.max())
n_tagged = df[df.has_genre].country.value_counts()
axes[2, 2].set(xticks=range(mix.shape[1]), yticks=range(mix.shape[0]),
               yticklabels=[f"{c} (n={n_tagged[c]})" for c in mix.index],
               title="Primary genre share per country")
axes[2, 2].set_xticklabels(mix.columns, rotation=45, ha="right")
axes[2, 2].grid(False)
for (i, j), v in pd.DataFrame(mix.values).stack().items():
    axes[2, 2].text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if v > mix.values.max() * 0.6 else "#0b0b0b")

fig.tight_layout()
fig.savefig(FIG_DIR / "table_overview.png", dpi=120)
plt.show()

# %% [markdown]
# ## Artist network
# Only artists with genre tags are included. Edge weight is the Jaccard
# similarity of the two artists' genre sets; edges below the threshold are
# dropped to avoid everything connecting through "rock"/"pop".

# %%
artist_genres = {a: set(g.split(";")) for a, g in artists.genres.dropna().items()}
print("Artists by number of genre tags:",
      Counter(len(t) for t in artist_genres.values()).most_common(6))

G_artist = nx.Graph()
for a in artist_genres:
    G_artist.add_node(a, songs=int(artists.at[a, "songs"]), country=artists.at[a, "country"])
for a, b in combinations(artist_genres, 2):
    ga, gb = artist_genres[a], artist_genres[b]
    jaccard = len(ga & gb) / len(ga | gb)
    if jaccard >= JACCARD_THRESHOLD:
        G_artist.add_edge(a, b, weight=jaccard)


def describe_graph(G, name):
    degrees = [d for _, d in G.degree()]
    components = list(nx.connected_components(G))
    giant = G.subgraph(max(components, key=len))
    print(f"--- {name} ---")
    print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}, "
          f"density: {nx.density(G):.4f}")
    print(f"Isolated nodes: {nx.number_of_isolates(G)}, components: {len(components)}, "
          f"giant component: {giant.number_of_nodes()} nodes")
    print(f"Degree mean/median/max: {sum(degrees) / len(degrees):.1f} / "
          f"{sorted(degrees)[len(degrees) // 2]} / {max(degrees)}")
    print(f"Avg clustering: {nx.average_clustering(G):.3f}")
    return giant


giant_artist = describe_graph(G_artist, "Artist network")
print("\nTop artists by degree:")
print(sorted(G_artist.degree, key=lambda x: -x[1])[:10])

# %%
def louvain(G, name):
    comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    comms = sorted(comms, key=len, reverse=True)
    q = nx.community.modularity(G, comms, weight="weight")
    print(f"{name}: {len(comms)} communities, modularity {q:.3f}")
    print(f"Sizes: {[len(c) for c in comms[:15]]}{' ...' if len(comms) > 15 else ''}")
    return comms


artist_comms = louvain(giant_artist, "Artist network (giant component)")

# characterise each community by the genres its artists are most often tagged with
for i, comm in enumerate(artist_comms[:8]):
    tags = Counter(g for a in comm for g in artist_genres[a])
    top_artists = sorted(comm, key=lambda a: -artists.at[a, "songs"])[:4]
    print(f"\n[{i}] {len(comm)} artists | genres: {', '.join(g for g, _ in tags.most_common(5))}")
    print(f"     e.g. {', '.join(top_artists)}")

# %%
def draw_communities(G, comms, title, path, size_attr, label_top=0, k=None):
    node_comm = {n: i for i, c in enumerate(comms) for n in c}
    cmap = plt.get_cmap("tab10")
    colors = [cmap(node_comm[n] % 10) for n in G]
    size = nx.get_node_attributes(G, size_attr)
    max_size = max(size.values())
    sizes = [15 + 600 * size[n] / max_size for n in G]
    pos = nx.spring_layout(G, weight="weight", seed=SEED, k=k)

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.05)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes, linewidths=0)
    if label_top:
        top = sorted(G, key=lambda n: -size[n])[:label_top]
        nx.draw_networkx_labels(G, pos, {n: n for n in top}, ax=ax, font_size=8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.show()


def save_interactive(G, comms, path, size_attr, tooltip, k=None):
    """Interactive HTML (pyvis) with the spring layout precomputed, so it opens
    instantly and matches the static figure. Hover a node for details; the
    filter menu can isolate a single community."""
    node_comm = {n: i for i, c in enumerate(comms) for n in c}
    cmap = plt.get_cmap("tab10")
    size = nx.get_node_attributes(G, size_attr)
    max_size = max(size.values())
    max_weight = max(w for *_, w in G.edges(data="weight"))
    pos = nx.spring_layout(G, weight="weight", seed=SEED, k=k)

    net = Network(height="900px", width="100%", cdn_resources="remote",
                  select_menu=True, filter_menu=True)
    for n in G:
        x, y = pos[n]
        net.add_node(n, label=n, title=tooltip(n, node_comm[n]), community=node_comm[n],
                     color=to_hex(cmap(node_comm[n] % 10)),
                     size=5 + 30 * size[n] / max_size, x=1500 * x, y=1500 * y)
    for a, b, w in G.edges(data="weight"):
        net.add_edge(a, b, value=w / max_weight, title=f"weight: {w:.2f}", color="#cccccc")
    net.toggle_physics(False)
    net.set_edge_smooth("false")
    net.write_html(str(path), notebook=False)
    print(f"Saved {path}")


draw_communities(giant_artist, artist_comms,
                 f"Artist network, giant component (Jaccard >= {JACCARD_THRESHOLD}), "
                 f"{len(artist_comms)} Louvain communities",
                 FIG_DIR / "artist_network.png", size_attr="songs", label_top=25)
save_interactive(
    giant_artist, artist_comms, FIG_DIR / "artist_network.html", size_attr="songs",
    tooltip=lambda a, c: (f"{a}\ncommunity: {c}\nsongs: {artists.at[a, 'songs']}\n"
                          f"country: {artists.at[a, 'country']}\n"
                          f"genres: {', '.join(sorted(artist_genres[a]))}"),
)

# %% [markdown]
# ## Genre network
# Nodes are genres, edge weight = number of artists tagged with both. Counted
# per artist rather than per song so prolific artists don't dominate.

# %%
G_genre = nx.Graph()
artist_tag_counts = Counter(g for tags in artist_genres.values() for g in tags)
for g, n in sorted(artist_tag_counts.items()):  # sorted so Louvain is reproducible
    G_genre.add_node(g, artists=n)
for tags in artist_genres.values():
    for a, b in combinations(sorted(tags), 2):
        w = G_genre.get_edge_data(a, b, {"weight": 0})["weight"]
        G_genre.add_edge(a, b, weight=w + 1)

giant_genre = describe_graph(G_genre, "Genre network")
print("\nStrongest genre pairs:")
for a, b, w in sorted(G_genre.edges(data="weight"), key=lambda e: -e[2])[:10]:
    print(f"  {a} -- {b}: {w}")

# %%
genre_comms = louvain(giant_genre, "Genre network (giant component)")
for i, comm in enumerate(genre_comms[:10]):
    top = sorted(comm, key=lambda g: -artist_tag_counts[g])[:8]
    print(f"[{i}] {len(comm)} genres: {', '.join(top)}")

draw_communities(giant_genre, genre_comms,
                 f"Genre co-occurrence network, {len(genre_comms)} Louvain communities",
                 FIG_DIR / "genre_network.png", size_attr="artists", label_top=35, k=0.5)
save_interactive(
    giant_genre, genre_comms, FIG_DIR / "genre_network.html", size_attr="artists", k=0.5,
    tooltip=lambda g, c: f"{g}\ncommunity: {c}\nartists: {artist_tag_counts[g]}",
)

# %% [markdown]
# ## Do artist communities line up with country?
# Share of each community's artists coming from its most common country.

# %%
rows = []
for i, comm in enumerate(artist_comms[:8]):
    countries = artists.loc[list(comm), "country"].value_counts()
    rows.append({"community": i, "artists": len(comm), "top_country": countries.index[0],
                 "top_country_share": round(countries.iloc[0] / len(comm), 2)})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nBaseline US share among tagged artists: "
      f"{(artists[artists.has_genre].country == 'US').mean():.2f}")
