# %% [markdown]
# # Exploration of the cleaned songs_1970_lyrics.csv
# Output of dataCleaner/clean.py + combine.py: one row per MusicBrainz song
# (duplicate recordings merged), with Genius lyrics attached where a match was found.
# - what the cleaning changed compared to songs_1970.csv
# - which songs got lyrics, and how much data has genres and/or lyrics
# - do the MusicBrainz genres agree with the Genius tag?
# - the lyrics themselves: length, repetition, distinctive words per Genius tag

# %%
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running in a notebook / interactive window
    HERE = Path.cwd()
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "songs_1970_lyrics.csv").exists())
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# same title/artist normalisation as the cleaning step
sys.path.insert(0, str(ROOT / "dataCleaner"))
from clean import artist_key, artist_main, title_core, title_key  # noqa: E402

df = pd.read_csv(ROOT / "songs_1970_lyrics.csv")
raw = pd.read_csv(ROOT / "songs_1970.csv")
genius = pd.read_csv(ROOT / "genius_1970_clean.csv", usecols=["title", "artist", "tag"])
TAGS = df.genius_tag.value_counts().index  # Genius tags, most common first
df.head()

# %% [markdown]
# ## Table overview

# %%
print(f"Shape: {df.shape}")
overview = pd.DataFrame({
    "nan": df.isna().sum(),
    "nan_%": (df.isna().mean() * 100).round(1),
    "unique": df.nunique(),
})
print(overview, "\n")

for col in ["year", "language"]:
    print(f"{col}: {df[col].unique()}")
# `country` is the kept recording's country, `countries` all merged release countries
print(f"Songs released in several countries: {df.countries.str.contains(';').sum()}")
print(f"Duplicate ids: {df.id.duplicated().sum()}, "
      f"duplicate (song, artist) pairs: {df.duplicated(['song', 'artist']).sum()}")

# %% [markdown]
# ### What did the cleaning change?
# clean.py merged recordings of the same song (country releases, mono/single
# versions, Part 1 / Part 2, credit variants). The artist column itself is left
# as is, so one act can still appear under several names, which matters for an
# artist network.

# %%
print(f"Recordings -> songs: {len(raw)} -> {len(df)} ({len(raw) - len(df)} merged)")
print(f"Artist names: {raw.artist.nunique()} -> {df.artist.nunique()}")
print(f"Songs without genres: {raw.genres.isna().mean():.0%} -> {df.genres.isna().mean():.0%}")

credits = df[["artist"]].drop_duplicates().assign(
    key=lambda d: d.artist.map(artist_key), main=lambda d: d.artist.map(artist_main))
print(f"\nArtist names: {len(credits)}, after normalising spelling: {credits.key.nunique()}, "
      f"main artist only: {credits.main.nunique()}")
variants = credits.groupby("main").filter(lambda g: g.key.nunique() > 1)
print(f"Main artists credited in several ways: {variants.main.nunique()}, e.g.")
for names in variants.groupby("main").artist.agg(" | ".join).head(10):
    print(f"  {names}")

# artist-level stats below use the spelling-normalised name, which only merges
# "Bob Marley & The Wailers" / "Bob Marley and The Wailers" style variants
df["artist_key"] = df.artist.map(artist_key)
df["has_genre"] = df.genres.notna()
df["has_lyrics"] = df.lyrics.notna()
df["has_both"] = df.has_genre & df.has_lyrics

# %% [markdown]
# ## Lyrics coverage
# ### Which matches are not exact?
# combine.py tries normalised title + artist first, then looser rules. The
# looser matches are printed to eyeball, plus the songs whose lyrics come from
# a Genius version page (live, remaster, ...) because Genius had no original.

# %%
print(df.match.value_counts().to_string(), "\n")
loose = df[df.has_lyrics & df.match.ne("title+artist")]
print(loose[["song", "artist", "genius_title", "genius_artist", "match"]].to_string(index=False))

genius_titles = df.genius_title.dropna()
version_pages = genius_titles[genius_titles.map(title_core) != genius_titles.map(title_key)]
print(f"\nLyrics taken from a version page: {len(version_pages)}")
print(", ".join(version_pages))

# %% [markdown]
# ### How much data do we have?
# The analyses need genres (MusicBrainz tags), lyrics (Genius), or both.

# %%
artists = df.groupby("artist_key").agg(
    songs=("id", "size"), has_genre=("has_genre", "any"),
    has_lyrics=("has_lyrics", "any"), has_both=("has_both", "any"))
cols = ["has_genre", "has_lyrics", "has_both"]
print(pd.DataFrame({
    "songs": [len(df), *df[cols].sum()],
    "artists": [len(artists), *artists[cols].sum()],
}, index=["all", "genres", "lyrics", "genres + lyrics"]))

# the Genius side: most 1970 Genius songs are not in the MusicBrainz sample
print(f"\nGenius songs from 1970 (cleaned): {len(genius)}, matched to a MusicBrainz song: "
      f"{df.has_lyrics.sum()} ({df.has_lyrics.sum() / len(genius):.0%})")

# %% [markdown]
# ## MusicBrainz genres vs. the Genius tag
# Genius has one coarse tag per song (pop, rock, rb, country), MusicBrainz a list
# of fine-grained artist tags. For songs with both: does the MusicBrainz list
# contain a genre matching the Genius tag? (see Overvejelser.md)

# %%
# MusicBrainz genres counted as agreeing with each Genius tag (regex on `genres`)
TAG_GENRES = {
    "pop": r"pop",
    "rock": r"rock|metal|punk",
    "rb": r"r&b|soul|funk|motown|rhythm and blues|doo-wop",
    "country": r"country|bluegrass|americana",
}
both = df[df.has_both].copy()
both["agree"] = [bool(re.search(TAG_GENRES.get(t, t), g))
                 for t, g in zip(both.genius_tag, both.genres)]
agreement = both.groupby("genius_tag").agree.agg(songs="size", agree="mean").loc[TAGS]
print(agreement.round(2))
print(f"Overall agreement: {both.agree.mean():.0%}\n")
print(both[~both.agree][["song", "artist", "genius_tag", "genres"]].head(12).to_string(index=False))

# songs with lyrics but no MusicBrainz genres could fall back on the Genius tag
print("\nGenius tag of songs without MusicBrainz genres:",
      df[df.has_lyrics & ~df.has_genre].genius_tag.value_counts().to_dict())

# %%
BLUE, GREY = "#2a78d6", "#c9c8c3"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.axisbelow": True})
overall = df.has_lyrics.mean()


def plot_lyrics_rate(ax, groups, order, title):
    """Share of songs with lyrics per group (n in the labels), against the overall rate."""
    rate = df.groupby(groups, observed=True).has_lyrics.agg(["mean", "size"]).loc[order]
    rate.index = [f"{g} (n={n})" for g, n in zip(rate.index, rate["size"])]
    rate["mean"].iloc[::-1].plot.barh(ax=ax, color=BLUE, width=0.7, xlim=(0, 1), title=title)
    ax.axvline(overall, color="#52514e", lw=1, ls="--")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("")


fig, axes = plt.subplots(2, 3, figsize=(18, 11))

plot_lyrics_rate(axes[0, 0], "country", df.country.value_counts().index[:10],
                 f"Songs with lyrics by country (overall {overall:.0%})")
plot_lyrics_rate(axes[0, 1], df.genre.fillna("(no genres)"),
                 df.genre.fillna("(no genres)").value_counts().index[:12],
                 "Songs with lyrics by primary genre")
artist_size = pd.cut(df.artist_key.map(artists.songs), [0, 1, 2, 5, 100],
                     labels=["1 song", "2 songs", "3-5 songs", "6+ songs"])
plot_lyrics_rate(axes[0, 2], artist_size, artist_size.cat.categories,
                 "Songs with lyrics by artist's songs in the data")

# which Genius tag each primary genre gets, rows normalised to shares
genre_tag = pd.crosstab(both.genre, both.genius_tag, normalize="index")
top_genres = both.genre.value_counts()[:10]
genre_tag = genre_tag.loc[top_genres.index, TAGS]
axes[1, 0].imshow(genre_tag.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
axes[1, 0].set(xticks=range(genre_tag.shape[1]), yticks=range(genre_tag.shape[0]),
               xticklabels=genre_tag.columns,
               yticklabels=[f"{g} (n={n})" for g, n in top_genres.items()],
               title="Genius tag per MusicBrainz primary genre")
axes[1, 0].grid(False)
for (i, j), v in pd.DataFrame(genre_tag.values).stack().items():
    axes[1, 0].text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else "#0b0b0b")

agree_rate = agreement.agree.copy()
agree_rate.index = [f"{t} (n={n})" for t, n in agreement.songs.items()]
agree_rate.iloc[::-1].plot.barh(ax=axes[1, 1], color=BLUE, width=0.4, xlim=(0, 1),
                                title=f"MusicBrainz genres agree with Genius tag "
                                      f"(overall {both.agree.mean():.0%})")
axes[1, 1].xaxis.set_major_formatter(PercentFormatter(1))

tag_shares = pd.DataFrame({
    f"all Genius 1970 songs (n={len(genius)})": genius.tag.value_counts(normalize=True),
    f"matched to MusicBrainz (n={df.has_lyrics.sum()})": df.genius_tag.value_counts(normalize=True),
}).reindex(genius.tag.value_counts().index).fillna(0)
tag_shares.plot.bar(ax=axes[1, 2], color=[GREY, BLUE], width=0.8, rot=0, xlabel="",
                    title="Genius tag share")
axes[1, 2].yaxis.set_major_formatter(PercentFormatter(1))

fig.tight_layout()
fig.savefig(FIG_DIR / "clean_coverage.png", dpi=120)
plt.show()

# %% [markdown]
# ## The lyrics
# Tokens are lowercase words, keeping inner apostrophes ("don't"). Repeated lines
# is the share of non-empty lines that repeat an earlier line, mostly the chorus.

# %%
lyr = df[df.has_lyrics].copy()
tokens = lyr.lyrics.str.lower().str.findall(r"[a-z]+(?:'[a-z]+)?")
lines = lyr.lyrics.str.lower().str.split("\n").map(lambda ls: [l.strip() for l in ls if l.strip()])
lyr["words"] = tokens.str.len()
lyr["distinct_words"] = tokens.map(lambda t: len(set(t)))
lyr["repeated_lines"] = lines.map(lambda ls: 1 - len(set(ls)) / len(ls))

print(f"Words per song: {lyr.words.describe().round(1).to_dict()}")
print(lyr.groupby("genius_tag")[["words", "distinct_words", "repeated_lines"]]
      .median().loc[TAGS].round(2))

# very short lyrics may be instrumentals with a spoken line or partial transcriptions
print("\nShortest lyrics:")
print(lyr.nsmallest(8, "words")[["song", "artist", "genius_title", "words"]].to_string(index=False))
print(f"\nMatched lyrics with no language set on Genius: {lyr.lyrics_language.isna().sum()}")

# %% [markdown]
# ### Distinctive words per Genius tag
# Share of a tag's songs that use a word, divided by the share of the other tags'
# songs that use it (+1 smoothing). Counted per song rather than per occurrence,
# so one chorus repeating a word doesn't dominate, and a word has to appear in at
# least MIN_SONG_SHARE of the tag's songs.

# %%
MIN_SONG_SHARE = 0.05

song_words = pd.DataFrame({"tag": lyr.genius_tag, "word": tokens.map(set)}).explode("word")
songs_using = pd.crosstab(song_words.word, song_words.tag)
n_songs = lyr.genius_tag.value_counts()
distinctive = {}
for t in TAGS:
    rest = songs_using.drop(columns=t).sum(axis=1)
    ratio = (songs_using[t] / n_songs[t]) / ((rest + 1) / (n_songs.drop(t).sum() + 2))
    distinctive[t] = ratio[songs_using[t] >= max(5, MIN_SONG_SHARE * n_songs[t])].nlargest(12)
    print(f"{t} ({n_songs[t]} songs): {', '.join(distinctive[t].index)}")

# %%
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].hist(lyr.words, bins=30, color=BLUE, rwidth=0.85)
axes[0, 0].set(title=f"Words per song ({len(lyr)} songs with lyrics)",
               xlabel="words", ylabel="songs")

box_style = dict(widths=0.5, medianprops={"color": BLUE, "lw": 2},
                 flierprops={"marker": "o", "markersize": 3, "markeredgecolor": "#52514e"})
for ax, col, title in zip(axes[0, 1:], ["words", "distinct_words", "repeated_lines"],
                          ["Words per song", "Distinct words per song", "Share of repeated lines"]):
    ax.boxplot([lyr.loc[lyr.genius_tag == t, col] for t in TAGS],
               tick_labels=[f"{t}\n(n={n_songs[t]})" for t in TAGS], **box_style)
    ax.set_title(f"{title} by Genius tag")
axes[0, 3].yaxis.set_major_formatter(PercentFormatter(1))

for ax, t in zip(axes[1], TAGS):
    distinctive[t].iloc[::-1].plot.barh(ax=ax, color=BLUE, width=0.7,
                                        title=f"Distinctive words: {t}")
    ax.set(xlabel="song share vs. other tags (x times)", ylabel="")

fig.tight_layout()
fig.savefig(FIG_DIR / "clean_lyrics.png", dpi=120)
plt.show()
