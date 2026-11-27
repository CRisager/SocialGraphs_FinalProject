"""
Download the Genius lyrics dataset from Kaggle and keep the songs from the years set in
src/config.py, written to data/raw/genius_songs_trimmed.csv.

Usage:
    python -m src.download.genius
"""

import os
import shutil

import kagglehub
import pandas as pd

from ..config import FIRST_YEAR, GENIUS_RAW, LAST_YEAR

HANDLE = "carlosgdcj/genius-song-lyrics-with-language-information"
COLUMNS = ["lyrics", "title", "artist", "year", "tag", "language"]


def main():
    # Downloads only song_lyrics.csv into the kagglehub cache and returns its path
    csv_path = kagglehub.dataset_download(HANDLE, path="song_lyrics.csv")
    df = pd.read_csv(csv_path, usecols=COLUMNS)
    df = df[df["year"].between(FIRST_YEAR, LAST_YEAR)]
    print(df["year"].value_counts().sort_index().to_string())

    GENIUS_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GENIUS_RAW, index=False)
    print(f"Wrote {GENIUS_RAW}")

    # Remove this dataset's cached copy
    cache_root = os.environ.get("KAGGLEHUB_CACHE", os.path.expanduser("~/.cache/kagglehub"))
    dataset_cache = os.path.join(cache_root, "datasets", *HANDLE.split("/"))
    shutil.rmtree(dataset_cache, ignore_errors=True)
    print(f"Removed cache: {dataset_cache}")


if __name__ == "__main__":
    main()
