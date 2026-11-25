"""
Data overview: how many songs survive each step, how the lyrics were matched, whether
MusicBrainz and Genius agree on the genre, and what the lyrics look like.

Reads the files the pipeline has already written (nothing is downloaded or re-cleaned).

Usage:
    python -m src.exploration        # prints the tables, writes figures/overview.png
                                     # and figures/lyrics.png
"""

import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .clean import SECTION_HEADER
from .combine import coarse_genre
from .config import (BLUE, FIG_DIR, FIRST_YEAR, GENIUS_RAW, GREY, LAST_YEAR, PLOT_STYLE,
                     TAG_COLORS, YEAR_RANGE, discogs_songs, musicbrainz_checkpoint,
                     musicbrainz_raw, SONGS_LYRICS)
from .data import load_data
from .text import tokens_for

TAGS = list(TAG_COLORS)  # Genius tags in a fixed order


# ---------------------------------------------------------------- tables

def musicbrainz_funnel(songs):
    """Songs left after each step, per year (columns) and in total."""
    # crawler checkpoints are only there for years crawled on this machine
    have_checkpoints = all(musicbrainz_checkpoint(y).exists() for y in YEAR_RANGE)
    rows = {}
    for y in YEAR_RANGE:
        row = {}
        if have_checkpoints:
            state = json.loads(musicbrainz_checkpoint(y).read_text())
            row["releases reported by API"] = state["search_total"]
            row["releases fetched"] = len(state["releases"])
        row["recordings first released that year"] = len(pd.read_csv(musicbrainz_raw(y)))
        rows[y] = row
    t = pd.DataFrame(rows).T
    both = songs.genius_tag.notna() & songs.mb_label.notna()
    for name, mask in [("songs after merging duplicates", songs.year.notna()),
                       ("with any MusicBrainz genre", songs.genre.notna()),
                       ("genre maps to a Genius tag", songs.mb_label.notna()),
                       ("with Genius lyrics", songs.match.notna()),
                       ("lyrics + both labels", both),
                       # Jonathan's row (jonathan branch): the songs the lyrics network
                       # and the classifier use
                       ("in the analysis (mb_label, no rap)",
                        songs.lyrics.notna() & songs.mb_label.isin(["rock", "rb", "pop", "country"])),
                       ("both labels agree", songs.labels_agree == True)]:  # noqa: E712
        t[name] = songs[mask].groupby("year").size().reindex(t.index, fill_value=0)
    t.loc["total"] = t.sum()
    return t.T.astype("Int64")


def genius_funnel(genius):
    """Genius songs left after each cleaning step, per year and in total."""
    g = pd.read_csv(GENIUS_RAW)
    g = g[g.year.between(FIRST_YEAR, LAST_YEAR)]
    steps = [("Genius pages in the year range", g)]
    g = g[g.tag != "misc"]
    steps.append(("not tagged 'misc'", g))
    g = g[g.language.eq("en") | g.language.isna()]
    steps.append(("English (or language unknown)", g))
    g = g[g.lyrics.str.replace(SECTION_HEADER, "", regex=True).str.strip() != ""]
    steps.append(("lyrics not empty", g))
    steps.append(("after merging versions/duplicates", genius))
    without_rap = genius[genius.tag != "rap"]
    steps.append(("without rap (used in the analyses)", without_rap))
    steps.append(("without rap and pop", without_rap[without_rap.tag != "pop"]))
    t = pd.DataFrame({name: d.groupby("year").size() for name, d in steps}).fillna(0)
    t.loc["total"] = t.sum()
    return t.T.astype(int)


def credit_coverage(year=FIRST_YEAR):
    """Credit coverage of one year's songs on MusicBrainz vs Discogs, per genre (the
    mb_label scheme without rap). Rates per genre:
      matched     share found on Discogs (MusicBrainz: all, they come from there)
      instrument  share with >= 1 instrument credit. MusicBrainz falls back from the
                  recording to its releases; for Discogs both track-level and
                  track-or-album are shown
      producer    share with >= 1 producer credit (MusicBrainz: only once the crawler
                  has been re-run with the credits phase)
      over-rep.   the genre's share of instrument-credited songs / its share of songs
                  (1 = proportional, > 1 = over-represented)
    Reads data/raw/songs_<year>.csv and, if there, data/raw/discogs_<year>.csv.
    Returns None when songs_<year>.csv was crawled without credits."""
    def has(col):
        return col.fillna("").astype(str).str.len() > 0

    labels = [t for t in TAGS if t != "rap"]
    mb = pd.read_csv(musicbrainz_raw(year))
    if "instruments" not in mb:
        return None
    songs = pd.DataFrame({"id": mb.id,
                          # current labels (combine.py); duplicates merged away get none
                          "genre": mb.id.map(pd.read_csv(SONGS_LYRICS, usecols=["id", "mb_label"])
                                             .set_index("id").mb_label),
                          "mb instrument": has(mb.instruments)})
    if "producers" in mb:
        songs["mb producer"] = has(mb.producers)
    if discogs_songs(year).exists():
        dc = pd.read_csv(discogs_songs(year)).set_index("id").reindex(songs.id)
        songs["discogs matched"] = has(dc["match"]).to_numpy()
        songs["discogs instrument (track)"] = has(dc.instruments).to_numpy()
        songs["discogs instrument (track or album)"] = \
            (has(dc.instruments) | has(dc.album_instruments)).to_numpy()
        songs["discogs producer (track or album)"] = \
            (has(dc.producers) | has(dc.album_producers)).to_numpy()

    songs = songs[songs.genre.isin(labels)]
    flags = [c for c in songs.columns if c not in ("id", "genre")]
    by = songs.groupby("genre")
    table = pd.DataFrame({"songs": by.size()}).reindex(labels)
    table.loc["all"] = len(songs)
    table["share"] = table.songs / len(songs)
    for col in flags:
        table[col] = by[col].mean().reindex(labels).tolist() + [songs[col].mean()]
        if "instrument" in col:
            credited = by[col].sum().reindex(labels)
            ratio = (credited / credited.sum()) / table.share[labels]
            table[col + " over-rep."] = ratio.tolist() + [1.0]
    return table.T


def label_agreement(songs):
    """Genius tag (rows) vs MusicBrainz label (columns) for songs that have both."""
    both = songs.genius_tag.notna() & songs.mb_label.notna()
    table = pd.crosstab(songs.genius_tag[both], songs.mb_label[both])
    order = [t for t in TAGS if t in table.index or t in table.columns]
    return table.reindex(index=order, columns=order, fill_value=0)


# MusicBrainz's own genre name for each Genius tag (the bars in overview.png)
TAG_GENRE_NAME = {"rock": "rock", "pop": "pop", "rb": "r&b", "country": "country",
                  "rap": "hip hop"}


def class_members(songs, n=15):
    """Which primary genres are combined into each coarse class (step 2 of label_steps).

    One row per class: its songs, how many distinct primary genres it combines and the
    n largest of them with their song counts. Genius tags first, then the other classes."""
    counts = songs.genre.value_counts()
    rows = {}
    for cls, members in counts.groupby(counts.index.map(coarse_genre)):
        shown = ", ".join(f"{g} ({k:,})" for g, k in members.head(n).items())
        if len(members) > n:
            shown += f", ... +{len(members) - n} more"
        rows[cls] = {"songs": members.sum(), "genres": len(members), "primary genres": shown}
    table = pd.DataFrame(rows).T
    others = table.drop(index=TAGS, errors="ignore").sort_values("songs", ascending=False)
    return pd.concat([table.reindex([t for t in TAGS if t in table.index]), others])


def label_steps(songs):
    """Songs per genre after each step from the MusicBrainz primary genre to mb_label.

    1. primary genre is literally the tag's name ("r&b", not "soul"), as in overview.png
    2. the primary genre's coarse class (soul, funk, disco -> rb; punk, wave -> rock ...)
    3. genre_coarse: the class with most votes, summed over all of a song's genres
    4. mb_label: as 3, but only the five Genius tags compete (jazz, folk ... drop out)
    5. mb_label and Genius lyrics found
    6. the analysis set of the lyrics network and classifier (rap dropped)
    """
    primary_class = songs.genre.map(coarse_genre, na_action="ignore")
    lyrics = songs.lyrics.notna()
    steps = {
        "1. primary genre\n= tag name": songs.genre.map(
            {name: tag for tag, name in TAG_GENRE_NAME.items()}),
        "2. primary genre\n→ its class": primary_class,
        "3. votes over\nall genres": songs.genre_coarse,
        "4. mb_label\n(5 tags only)": songs.mb_label,
        "5. with Genius\nlyrics": songs.mb_label.where(lyrics),
        "6. in the analysis\n(no rap)": songs.mb_label.where(lyrics & (songs.mb_label != "rap")),
    }
    return (pd.DataFrame({step: labels.value_counts() for step, labels in steps.items()})
            .reindex(TAGS).fillna(0).astype(int))


def lyric_stats(songs):
    """Songs with lyrics, with word counts (all words), the share of repeated lines,
    and the filtered spaCy tokens (src/text.py) used for distinctive words."""
    lyr = songs[songs.lyrics.notna()].copy()
    all_words = lyr.lyrics.str.lower().str.findall(r"[a-z]+(?:'[a-z]+)?")
    lines = lyr.lyrics.str.lower().str.split("\n").map(
        lambda ls: [line.strip() for line in ls if line.strip()])
    lyr["tokens"] = tokens_for(lyr.lyrics)
    lyr["words"] = all_words.str.len()
    lyr["distinct_words"] = all_words.map(lambda t: len(set(t)))
    lyr["repeated_lines"] = lines.map(lambda ls: 1 - len(set(ls)) / len(ls) if ls else 0)
    return lyr


def distinctive_words(lyr, n=12, min_song_share=0.05, min_songs=100):
    """Per Genius tag: share of the tag's songs using a word divided by the share of the
    other tags' songs using it (+1 smoothing). Counted per song, so one chorus repeating
    a word doesn't dominate; a word must appear in >= min_song_share of the tag's songs.
    Tags with fewer than min_songs songs (rap) are skipped: their ratios are noise."""
    song_words = pd.DataFrame({"tag": lyr.genius_tag, "word": lyr.tokens.map(set)}
                              ).explode("word")
    using = song_words.groupby(["word", "tag"]).size().unstack(fill_value=0)
    n_songs = lyr.genius_tag.value_counts()
    result = {}
    for t in [t for t in TAGS if n_songs.get(t, 0) >= min_songs]:
        rest = using.drop(columns=t).sum(axis=1)
        ratio = (using[t] / n_songs[t]) / ((rest + 1) / (n_songs.drop(t).sum() + 2))
        result[t] = ratio[using[t] >= max(5, min_song_share * n_songs[t])].nlargest(n)
    return result


# ---------------------------------------------------------------- figures

def _barh(ax, values, title, fmt=None, color=BLUE):
    values.iloc[::-1].plot.barh(ax=ax, color=color, width=0.7)
    ax.set_title(title, loc="left")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    if fmt:
        ax.xaxis.set_major_formatter(fmt)


def plot_overview(songs, genius, path=FIG_DIR / "overview.png"):
    """What the MusicBrainz songs look like and where lyrics were found."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        per_year = pd.DataFrame({
            "with lyrics": songs[songs.lyrics.notna()].groupby("year").size(),
            "without lyrics": songs[songs.lyrics.isna()].groupby("year").size(),
        }).fillna(0)
        per_year.plot.bar(ax=axes[0, 0], stacked=True, color=[BLUE, GREY], width=0.8,
                          rot=45, edgecolor="white", linewidth=1)
        axes[0, 0].set(title="MusicBrainz songs per year", xlabel="")
        axes[0, 0].grid(axis="x", visible=False)

        _barh(axes[0, 1], songs.genre.value_counts().head(15),
              f"Top 15 primary genres ({songs.genre.isna().sum()} songs have none)")
        _barh(axes[0, 2], songs.country.value_counts().head(10), "Top 10 release countries")

        top = songs.genre.value_counts().index[:15]
        rate = songs.groupby("genre").lyrics.apply(lambda s: s.notna().mean()).loc[top]
        rate.index = [f"{g} (n={n})" for g, n in songs.genre.value_counts()[:15].items()]
        _barh(axes[1, 0], rate, "Share of songs with lyrics, per primary genre",
              PercentFormatter(1))
        axes[1, 0].axvline(songs.lyrics.notna().mean(), color="#52514e", lw=1, ls="--")

        shares = pd.DataFrame({
            f"all Genius songs (n={len(genius)})": genius.tag.value_counts(normalize=True),
            f"matched to MusicBrainz (n={songs.genius_tag.notna().sum()})":
                songs.genius_tag.value_counts(normalize=True),
        }).reindex(TAGS).fillna(0)
        shares.plot.bar(ax=axes[1, 1], color=[GREY, BLUE], width=0.8, rot=0,
                        edgecolor="white", linewidth=1)
        axes[1, 1].set(title="Genius tag share", xlabel="")
        axes[1, 1].legend(loc="upper right")
        axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, 1].grid(axis="x", visible=False)

        agree = label_agreement(songs)
        share = agree.div(agree.sum(axis=1), axis=0)
        axes[1, 2].imshow(share.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axes[1, 2].set(xticks=range(share.shape[1]), yticks=range(share.shape[0]),
                       xticklabels=share.columns,
                       yticklabels=[f"{t} (n={n})" for t, n in agree.sum(axis=1).items()],
                       xlabel="MusicBrainz label", ylabel="Genius tag",
                       title="Genius tag vs MusicBrainz label (row %)")
        axes[1, 2].grid(False)
        for (i, j), v in pd.DataFrame(share.values).stack().items():
            axes[1, 2].text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                            color="white" if v > 0.6 else "#0b0b0b")

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    return fig


def plot_label_steps(steps, path=FIG_DIR / "label_steps.png"):
    """One line per genre through the steps of label_steps(), every count printed."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(steps.shape[1])
        for tag, counts in steps.iterrows():
            shown = counts.replace(0, float("nan"))  # log scale: leave out empty steps
            ax.plot(x, shown, color=TAG_COLORS[tag], lw=2, marker="o", ms=8,
                    markeredgecolor="white", markeredgewidth=2, label=tag)
            for i, n in enumerate(shown):
                if n == n:  # not NaN
                    # below the point when another genre sits just above it and the
                    # space below is free, else above; white box keeps crossing lines off
                    col = steps.iloc[:, i]
                    below = (((col > n) & (col < 1.6 * n)).any()
                             and not ((col < n) & (col > n / 1.6)).any())
                    ax.annotate(f"{int(n):,}", (i, n), xytext=(0, -10 if below else 9),
                                textcoords="offset points", ha="center",
                                va="top" if below else "baseline", fontsize=9,
                                color="#0b0b0b", bbox=dict(boxstyle="square,pad=0.1",
                                                           fc="white", ec="none"))
            last = shown.reset_index(drop=True).last_valid_index()  # rap ends before step 6
            if last is not None:
                ax.annotate(tag, (last, shown.iloc[last]), xytext=(14, 0),
                            textcoords="offset points", va="center", fontsize=10,
                            color="#52514e")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set(xticks=list(x), xlim=(-0.3, len(x) - 0.4), ylabel="songs (log scale)",
               title="Songs per genre through the labeling steps")
        ax.set_xticklabels(steps.columns, fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="lower left", ncol=len(steps), frameon=False)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    return fig


def plot_lyrics(lyr, distinctive, path=FIG_DIR / "lyrics.png"):
    """Length and repetition of the matched lyrics, and distinctive words per tag."""
    tags = [t for t in TAGS if t in set(lyr.genius_tag)]
    n_songs = lyr.genius_tag.value_counts()
    cap = lyr.words.quantile(0.99)
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes[0, 0].hist(lyr.words[lyr.words <= cap], bins=40, color=BLUE, rwidth=0.85)
        axes[0, 0].set(title=f"Words per song ({len(lyr)} songs, longest 1% not shown)",
                       xlabel="words", ylabel="songs")
        for ax, col, title in zip(axes[0, 1:], ["words", "distinct_words", "repeated_lines"],
                                  ["Words per song", "Distinct words per song",
                                   "Share of repeated lines"]):
            parts = ax.boxplot([lyr.loc[lyr.genius_tag == t, col] for t in tags],
                               tick_labels=[f"{t}\n(n={n_songs[t]})" for t in tags],
                               widths=0.5, patch_artist=True, showfliers=False,
                               medianprops={"color": "#0b0b0b", "lw": 1.5})
            for box, t in zip(parts["boxes"], tags):
                box.set(facecolor=TAG_COLORS[t], edgecolor=TAG_COLORS[t], alpha=0.8)
            ax.set_title(f"{title} by Genius tag")
            ax.grid(axis="x", visible=False)
        axes[0, 3].yaxis.set_major_formatter(PercentFormatter(1))

        for ax, t in zip(axes[1], distinctive):
            _barh(ax, distinctive[t], f"Distinctive words: {t}", color=TAG_COLORS[t])
            ax.set_xlabel("x more songs use it than in other tags")
        for ax in axes[1, len(distinctive):]:
            ax.axis("off")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    return fig


def main():
    songs, genius = load_data()
    print("MusicBrainz side\n" + musicbrainz_funnel(songs).to_string())
    print("\nGenius side\n" + genius_funnel(genius).to_string())
    print("\nHow the lyrics were matched\n" + songs.match.value_counts().to_string())
    print("\nGenius tag vs MusicBrainz label\n" + label_agreement(songs).to_string())
    lyr = lyric_stats(songs)
    plot_overview(songs, genius)
    plot_lyrics(lyr, distinctive_words(lyr))
    print(f"\nWrote {FIG_DIR / 'overview.png'} and {FIG_DIR / 'lyrics.png'}")


if __name__ == "__main__":
    main()
