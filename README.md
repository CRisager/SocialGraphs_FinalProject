# SocialGraphs_FinalProject

English-language songs first released 1970–1980: MusicBrainz metadata, genres and
credits, matched to Genius lyrics. We ask whether songs with similar lyrics or shared
genres form communities, and whether a song's genre can be predicted from its lyrics.

**Start with [main.ipynb](main.ipynb).** Run it top to bottom; it shows every key result.
[Documentation.ipynb](Documentation.ipynb) describes the planned methods and why.

## Project layout

```
main.ipynb               results: data overview, genre network, lyrics network, classification
Documentation.ipynb      planned methods, step by step
src/
  config.py              year range, file paths, plot colours  <- change settings here
  download/
    musicbrainz.py       crawl MusicBrainz            -> data/raw/songs_<year>.csv, credits_<year>.csv
    genius.py            Genius lyrics from Kaggle    -> data/raw/genius_songs_trimmed.csv
    discogs.py           Discogs credits via the API  -> data/raw/discogs_<year>.csv, discogs_credits_<year>.csv
    discogs_dump.py      the same from a Discogs data dump (no API)
  clean.py               deduplicate both sources     -> data/processed/*_clean.csv
  combine.py             attach lyrics + labels       -> data/processed/songs_<years>_lyrics.csv
  data.py                load_data(): loads processed files, rebuilds them if missing
  text.py                spaCy tokenization of the lyrics (cached in data/processed/tokens.csv)
  exploration.py         data overview tables and plots, credit coverage
  genre_network.py       songs linked by shared genres, communities
  lyrics_network.py      songs/artists linked by similar lyrics, communities vs genres, null models
  classification.py      predict genre from lyrics
data/                    (not in git) raw/ downloads and processed/ outputs
figures/                 plots written by the analyses (the HTML ones are not in git)
results/                 result tables and collection_log.jsonl (one line per crawler run)
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Put the data in `data/raw/` (ask a teammate for the files, or download them, below).

## Running the pipeline

The notebook runs everything after the download. Each step can also be run on its own
from the repository root:

```bash
python -m src.download.musicbrainz     # resumable, see below
python -m src.download.genius          # needs at least 9 GB free disk space
python -m src.download.discogs --year 1970   # optional; needs DISCOGS_TOKEN
python -m src.clean                    # -> data/processed/songs_1970-1980_clean.csv, genius_1970-1980_clean.csv
python -m src.combine                  # -> data/processed/songs_1970-1980_lyrics.csv
python -m src.exploration              # tables + figures/overview.png, figures/lyrics.png
python -m src.genre_network            # figures/genre_network.html
python -m src.lyrics_network           # ~20 min; results/lyrics_network_*.csv + figures
python -m src.classification           # macro-F1 scores + confusion matrices
```

The year range is `FIRST_YEAR, LAST_YEAR` in [src/config.py](src/config.py).

## How the data is built

- **MusicBrainz** ([musicbrainz.py](src/download/musicbrainz.py)): English-language
  official singles; one row per recording first released in the year, with genres
  (the recording's tags, else its artists') and credits: `instruments`, `producers`,
  `musicians`, `vocals` and record `labels` (empty where MusicBrainz has none). Every
  single credit, with MusicBrainz ids for building a collaboration graph, goes to
  `credits_<year>.csv`. The credits phase needs one request per release, about 5.5 hours
  for 1970–1980; add `--no-credits` to skip it. The crawler may get timed out if run too
  often, as it respects the rate limits. Each run appends its endpoint, query, dates,
  request count and resulting counts to `results/collection_log.jsonl` for the Methods
  section.
- **Discogs** ([discogs.py](src/download/discogs.py), [discogs_dump.py](src/download/discogs_dump.py)):
  the same songs looked up on Discogs, to compare credit coverage
  (`credit_coverage()` in [exploration.py](src/exploration.py)).
- **Cleaning** ([clean.py](src/clean.py)): recordings of the same song (country
  releases, mono/single versions, Part 1/2, credit variants) are merged, combining their
  genres, credits and countries. Genius non-song pages, non-English lyrics and version
  pages (live, remaster, ...) are dropped. The primary `genre` is reduced to a
  single-word core genre: the rightmost recognised one wins (`pop rock` -> `rock`); the
  full list stays in `genres`.
- **Matching** ([combine.py](src/combine.py)): exact title + artist first, then looser
  rules, then fuzzy titles within the same artist. MusicBrainz genres are mapped onto
  the Genius tags (pop, rock, rb, country, rap) as `mb_label`, which is the genre used
  by the lyrics network and the classifier (rap dropped: too few songs); `label` is the
  Genius tag where there is one, and `labels_agree` marks songs where both sources agree.

## Tokenization ([text.py](src/text.py))

Lyrics are lowercased with `str.lower()` and tokenized with spaCy's English tokenizer
(`en_core_web_sm`). A token is kept only when

1. `token.is_alpha` is true, so punctuation, symbols, numbers and alphanumeric tokens
   are removed;
2. `token.is_stop` is false (spaCy's English stop-word list);
3. it is not one of the section markers `verse`, `chorus`, `bridge`, `intro`, `outro`,
   `hook`, `refrain`.

Words that occur in only one song are then removed by the TF-IDF (`min_df=2`): a word
must appear in at least two distinct songs, counting each song once. Only spaCy's
tokenizer runs (no tagger, parser or NER): `is_alpha` and `is_stop` are properties of the
word itself, so the tokens are identical to the full pipeline's, at ~400 songs/s.
Tokens are cached in `data/processed/tokens.csv`, keyed by a hash of the lyrics.
