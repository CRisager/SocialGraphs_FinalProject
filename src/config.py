"""Project-wide settings: the year range and where every file lives.

    data/raw/        downloads: songs_<year>.csv, mb_<year>_checkpoint.json,
                     genius_songs_trimmed.csv
    data/processed/  written by src/clean.py and src/combine.py
    figures/         plots and interactive HTML written by the analysis modules
    results/         result tables and the data collection log
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"  # result tables (small CSVs, kept in git)

# Years to use. Every step after the download reads the files named by YEARS.
FIRST_YEAR, LAST_YEAR = 1970, 1980
YEARS = str(FIRST_YEAR) if FIRST_YEAR == LAST_YEAR else f"{FIRST_YEAR}-{LAST_YEAR}"
YEAR_RANGE = range(FIRST_YEAR, LAST_YEAR + 1)

SEED = 42

# raw downloads
GENIUS_RAW = RAW_DIR / "genius_songs_trimmed.csv"


def musicbrainz_raw(year):
    return RAW_DIR / f"songs_{year}.csv"


def musicbrainz_checkpoint(year):
    return RAW_DIR / f"mb_{year}_checkpoint.json"


def musicbrainz_credits(year):
    """One row per credit (producer, musician, singer, label) of a year's songs."""
    return RAW_DIR / f"credits_{year}.csv"


def discogs_songs(year):
    return RAW_DIR / f"discogs_{year}.csv"


def discogs_credits(year):
    return RAW_DIR / f"discogs_credits_{year}.csv"


def discogs_checkpoint(year):
    return RAW_DIR / f"discogs_{year}_checkpoint.json"


# one line per crawler run (endpoint, query, dates, request counts) for the Methods
# section; kept in git, unlike the data
COLLECTION_LOG = RESULTS_DIR / "collection_log.jsonl"


# processed files
SONGS_CLEAN = PROCESSED_DIR / f"songs_{YEARS}_clean.csv"    # deduplicated MusicBrainz songs
GENIUS_CLEAN = PROCESSED_DIR / f"genius_{YEARS}_clean.csv"  # cleaned Genius songs
SONGS_LYRICS = PROCESSED_DIR / f"songs_{YEARS}_lyrics.csv"  # MusicBrainz songs + lyrics

# plot colours (validated reference palette); one fixed colour per Genius tag so a
# genre looks the same in every figure
BLUE, GREY = "#2a78d6", "#c9c8c3"
TAG_COLORS = {"rock": "#2a78d6", "pop": "#eb6834", "rb": "#1baf7a",
              "country": "#eda100", "rap": "#e87ba4"}
PLOT_STYLE = {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
              "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
              "axes.titlesize": 12, "axes.titleweight": "bold",
              "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
              "xtick.color": "#52514e", "ytick.color": "#52514e"}
