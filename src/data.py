"""Load the processed datasets, rebuilding them from data/raw/ when needed."""

import pandas as pd

from . import clean, combine
from .config import GENIUS_CLEAN, GENIUS_RAW, SONGS_LYRICS, YEAR_RANGE, musicbrainz_raw


def _stale():
    """Why the processed files must be rebuilt, or None if they are up to date: one is
    missing, or a raw download is newer (e.g. new songs_<year>.csv files were added)."""
    processed = [GENIUS_CLEAN, SONGS_LYRICS]
    if not all(p.exists() for p in processed):
        return "processed files missing"
    built = min(p.stat().st_mtime for p in processed)
    newer = [r.name for r in [GENIUS_RAW, *map(musicbrainz_raw, YEAR_RANGE)]
             if r.exists() and r.stat().st_mtime > built]
    return f"newer raw files: {', '.join(newer)}" if newer else None


def load_data(rebuild=False):
    """Return (songs, genius).

    songs   every MusicBrainz song, with Genius lyrics and labels where matched
            (songs_<years>_lyrics.csv)
    genius  every cleaned Genius song in the year range (genius_<years>_clean.csv)

    The cleaning and matching (a few minutes) only run when rebuild=True, a processed
    file is missing, or a raw file in data/raw/ is newer than the processed files.
    """
    reason = "rebuild=True" if rebuild else _stale()
    if reason:
        print(f"Rebuilding processed data ({reason})...")
        combine.run(*clean.run())
    # always read from disk, so dtypes are the same whether or not we rebuilt
    return pd.read_csv(SONGS_LYRICS), pd.read_csv(GENIUS_CLEAN)
