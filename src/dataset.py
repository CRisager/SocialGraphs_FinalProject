"""
The song table used by the network analysis (notebooks/analysis.ipynb).

Songs first released ANALYSIS_YEARS, one row per song (index = MusicBrainz id), with
  tokens        spaCy word tokens of the lyrics (src/text.py), without filler words when
                config.REMOVE_FILLER_WORDS is True; NaN without lyrics
  genre_eval    the evaluation genre: genre_coarse, with classes under MIN_GENRE_SIZE
                songs (counted among the songs with lyrics) merged into "other", so no
                class is too small to measure; None for songs without a genre
  has_lyrics, has_instruments
Genre is metadata only: it is never used to build an edge.

Duplicates. The cleaning merges recordings of one song by title and artist, but a few
spelling variants slip through ("Last Morning" / "Last Mornin'", "... Babe" /
"... Baby") and match two Genius pages with the same lyrics. Two songs are treated as
duplicates when their lyrics are near-identical (cosine >= DUPLICATE_COSINE) and their
artists are the same, or one artist name contains the other ("Cockney Rebel" /
"Steve Harley & Cockney Rebel"). Covers (the same lyrics by a different artist) are
kept: they are real lyrical links.
"""

import numpy as np
import pandas as pd

from .clean import artist_key
from .config import FIRST_YEAR, LAST_YEAR, REMOVE_FILLER_WORDS
from .data import load_data
from .text import drop_fillers, lyrics_vectorizer, tokens_for

ANALYSIS_YEARS = (FIRST_YEAR, LAST_YEAR)   # 1970-1980, set in src/config.py
MIN_GENRE_SIZE = 30
DUPLICATE_COSINE = 0.95


def _same_artist(a, b):
    return a == b or a in b or b in a


def lyric_duplicates(songs):
    """Ids of songs to drop as duplicates of another song (see module docstring). Of a
    pair we keep the song with instrument data, else the earlier one."""
    lyr = songs[songs.tokens.notna()]
    X = lyrics_vectorizer(min_df=2, max_df=0.5).fit_transform(lyr.tokens)
    S = (X @ X.T).tocoo()
    keys = lyr.artist.map(artist_key).to_numpy()
    ids = lyr.index.to_numpy()
    rank = list(zip(lyr.instruments.isna(), lyr.year, range(len(lyr))))  # lower = keep
    drop = set()
    for i, j, s in zip(S.row, S.col, S.data):
        if i < j and s >= DUPLICATE_COSINE and _same_artist(keys[i], keys[j]):
            drop.add(ids[j] if rank[i] < rank[j] else ids[i])
    return sorted(drop)


def eval_genre(coarse, has_lyrics, min_size=MIN_GENRE_SIZE):
    """genre_coarse with classes under min_size songs with lyrics merged into 'other'."""
    counts = coarse[has_lyrics].value_counts()
    small = set(counts[counts < min_size].index)
    return coarse.where(~coarse.isin(small), "other")


def load_analysis_songs(remove_fillers=REMOVE_FILLER_WORDS, verbose=True):
    songs, _ = load_data()
    s = songs[songs.year.between(*ANALYSIS_YEARS)].set_index("id")
    s = s.assign(has_lyrics=s.lyrics.notna(), has_instruments=s.instrument_families.notna())
    tokens = tokens_for(s.lyrics[s.has_lyrics], verbose=verbose)
    if remove_fillers:
        n_before = tokens.map(len).sum()
        tokens = tokens.map(drop_fillers)
        if verbose:
            print(f"Filler words removed: {1 - tokens.map(len).sum() / n_before:.1%} of the "
                  f"lyrics tokens (config.REMOVE_FILLER_WORDS)")
    s["tokens"] = tokens.reindex(s.index)
    dups = lyric_duplicates(s)
    s = s.drop(index=dups)
    s["genre_eval"] = eval_genre(s.genre_coarse, s.has_lyrics)
    if verbose:
        print(f"{len(s)} songs {ANALYSIS_YEARS[0]}-{ANALYSIS_YEARS[1]} "
              f"({len(dups)} lyric duplicates dropped): {s.has_lyrics.sum()} with lyrics, "
              f"{s.has_instruments.sum()} with instruments, "
              f"{(s.has_lyrics & s.has_instruments).sum()} with both")
    return s


assert _same_artist("cockneyrebel", "steveharleyandcockneyrebel")
assert not _same_artist("elvispresley", "brendalee")
_c = pd.Series(["rock"] * 40 + ["jazz"] * 5 + [None])
assert eval_genre(_c, pd.Series([True] * 46)).value_counts().to_dict() == {"rock": 40, "other": 5}
