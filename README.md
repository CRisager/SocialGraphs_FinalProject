# SocialGraphs_FinalProject

English-language songs first released 1970–1980: MusicBrainz metadata and genres, matched
to Genius lyrics. We ask whether songs with similar lyrics or shared genres form
communities, and whether a song's genre can be predicted from its lyrics.

**Start with [main.ipynb](main.ipynb).** Run it top to bottom; it shows every key result.

## Project layout

```
main.ipynb               results: data overview, genre network, lyrics network, classification
src/
  config.py              year range, file paths, plot colours  <- change settings here
  download/
    musicbrainz.py       crawl MusicBrainz            -> data/raw/songs_<year>.csv
    genius.py            Genius lyrics from Kaggle    -> data/raw/genius_songs_trimmed.csv
  clean.py               deduplicate both sources     -> data/processed/*_clean.csv
  combine.py             attach lyrics + labels       -> data/processed/songs_<years>_lyrics.csv
  data.py                load_data(): loads processed files, rebuilds them if missing
  exploration.py         data overview tables and plots
  genre_network.py       songs linked by shared genres, communities
  lyrics_network.py      songs linked by similar lyrics, communities vs genres
  classification.py      predict genre from lyrics
data/                    (not in git) raw/ downloads and processed/ outputs
figures/                 plots written by the analyses (the HTML ones are not in git)
```

## Setup

```bash
pip install -r requirements.txt
```

Put the data in `data/raw/` (ask a teammate for the files, or download them, below).

## Running the pipeline

The notebook runs everything after the download. Each step can also be run on its own
from the repository root:

```bash
python -m src.download.musicbrainz     # resumable; ~5.5 h for 1970-1980, add --no-instruments to skip the slow phase
python -m src.download.genius
python -m src.clean                    # -> data/processed/songs_1970-1980_clean.csv, genius_1970-1980_clean.csv
python -m src.combine                  # -> data/processed/songs_1970-1980_lyrics.csv
python -m src.exploration              # tables + figures/overview.png, figures/lyrics.png
python -m src.genre_network            # figures/genre_network.html
python -m src.lyrics_network           # figures/lyrics_network.png/.html
python -m src.classification           # macro-F1 scores + confusion matrices
```

The year range is `FIRST_YEAR, LAST_YEAR` in [src/config.py](src/config.py).

## How the data is built

- **MusicBrainz** ([musicbrainz.py](src/download/musicbrainz.py)): English-language
  official singles; one row per recording first released in the year, with genres
  (the recording's tags, else its artists') and credited instruments.
- **Cleaning** ([clean.py](src/clean.py)): recordings of the same song (country
  releases, mono/single versions, Part 1/2, credit variants) are merged. Genius
  non-song pages, non-English lyrics and version pages (live, remaster, ...) are dropped.
  The primary `genre` is reduced to a single-word core genre: the rightmost recognised
  one wins (`pop rock` -> `rock`); the full list stays in `genres`.
- **Matching** ([combine.py](src/combine.py)): exact title + artist first, then looser
  rules, then fuzzy titles within the same artist. MusicBrainz genres are mapped onto
  the Genius tags (pop, rock, rb, country, rap) as `mb_label`; `label` is the Genius tag
  where there is one, and `labels_agree` marks songs where both sources agree.
