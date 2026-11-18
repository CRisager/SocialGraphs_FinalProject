"""Load the processed datasets, rebuilding them from data/raw/ when needed."""

import pandas as pd

from . import clean, combine
from .config import GENIUS_CLEAN, SONGS_LYRICS


def load_data(rebuild=False):
    """Return (songs, genius).

    songs   every MusicBrainz song, with Genius lyrics and labels where matched
            (songs_<years>_lyrics.csv)
    genius  every cleaned Genius song in the year range (genius_<years>_clean.csv)

    The cleaning and matching (a few minutes) only run when a processed file is
    missing or rebuild=True.
    """
    if rebuild or not GENIUS_CLEAN.exists() or not SONGS_LYRICS.exists():
        combine.run(*clean.run())
    # always read from disk, so dtypes are the same whether or not we rebuilt
    return pd.read_csv(SONGS_LYRICS), pd.read_csv(GENIUS_CLEAN)
