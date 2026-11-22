# SocialGraphs_FinalProject

English-language songs first released 1970–1980: MusicBrainz metadata, genres and
credits, matched to Genius lyrics.

**Research question.** To what extent can songs from 1970–1975 be grouped into
meaningful clusters based on lyrical themes, country and instrumentation, and how do
these clusters correspond to established musical genres?

- [notebooks/analysis.ipynb](notebooks/analysis.ipynb): the network analysis for the
  research question (three song graphs, backbones, communities, null models), step by
  step. It is also our explainer notebook.
- [Documentation.ipynb](Documentation.ipynb): the methods table and every data
  decision with its reasoning (genre labels, country, instrument families, which songs
  enter which graph).
- [main.ipynb](main.ipynb): the data overview, the genre network, the earlier
  lyrics network and the genre classifier, run top to bottom.

## Project layout

```
notebooks/analysis.ipynb the network analysis for the research question (explainer notebook)
Documentation.ipynb      methods table and data decisions, with reasons
main.ipynb               data overview, genre network, lyrics network, classification
src/
  config.py              year range, file paths, SEED, REMOVE_FILLER_WORDS  <- change settings here
  download/
    musicbrainz.py       crawl MusicBrainz            -> data/raw/songs_<year>.csv, credits_<year>.csv
    musicbrainz_dump.py  fill in the credits from a MusicBrainz database dump (no API)
    genius.py            Genius lyrics from Kaggle    -> data/raw/genius_songs_trimmed.csv
    discogs.py           Discogs credits via the API  -> data/raw/discogs_<year>.csv, discogs_credits_<year>.csv
    discogs_dump.py      the same from a Discogs data dump (no API)
  clean.py               deduplicate both sources, genre votes -> data/processed/*_clean.csv
  instruments.py         instrument names -> 14 instrument families
  combine.py             attach lyrics + labels       -> data/processed/songs_<years>_lyrics.csv
  data.py                load_data(): loads processed files, rebuilds them if missing
  text.py                spaCy tokenization of the lyrics (cached in data/processed/tokens.csv),
                         filler words, TF-IDF vectorizer
  dataset.py             the 1970-1975 song table of the network analysis (genre_eval, tokens)
  graphs.py              G_lyrics, G_instruments, G_combined (TF-IDF, hyperbolic projections)
  backbone.py            disparity filter
  plotting.py            PNAS-sized figure style, PDF + PNG saving
  exploration.py         data overview tables and plots, credit coverage
  genre_network.py       songs linked by shared genres, communities
  lyrics_network.py      songs/artists linked by similar lyrics, communities vs genres, null models
                         (--balanced: equal songs per genre)
  classification.py      predict genre from lyrics (--excluded: leave words out)
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
python -m src.download.musicbrainz_dump <mbdump folder> --years 1970 1980  # credits from a dump, then rerun the line above
python -m src.download.genius          # needs at least 9 GB free disk space
python -m src.download.discogs --year 1970   # optional; needs DISCOGS_TOKEN
python -m src.clean                    # -> data/processed/songs_1970-1980_clean.csv, genius_1970-1980_clean.csv
python -m src.combine                  # -> data/processed/songs_1970-1980_lyrics.csv
python -m src.exploration              # tables + figures/overview.png, figures/lyrics.png
python -m src.genre_network            # figures/genre_network.html
python -m src.lyrics_network           # ~20 min; results/lyrics_network_*.csv + figures
python -m src.lyrics_network --balanced   # ~10 min; same on samples with equal songs per genre
python -m src.classification           # macro-F1 scores + confusion matrices
python -m src.classification --excluded love baby yeah   # the same without these words
```

The year range is `FIRST_YEAR, LAST_YEAR` in [src/config.py](src/config.py).

## How the data is built

- **MusicBrainz** ([musicbrainz.py](src/download/musicbrainz.py)): English-language
  official singles; one row per recording first released in the year, with genres
  (the recording's tags, else its artists') and credits: `instruments`, `producers`,
  `musicians`, `vocals` and record `labels` (empty where MusicBrainz has none). Every
  single credit, with MusicBrainz ids for building a collaboration graph, goes to
  `credits_<year>.csv`. The credits phase needs one request per release, about 5.5 hours
  for 1970–1980; add `--no-credits` to skip it. Instead,
  [musicbrainz_dump.py](src/download/musicbrainz_dump.py) can fill the credits in
  from a MusicBrainz database dump, after which the crawler only writes the CSVs (this is
  how our credits were made). The crawler may get timed out if run too
  often, as it respects the rate limits. Each run appends its endpoint, query, dates,
  request count and resulting counts to `results/collection_log.jsonl` for the Methods
  section.
- **Discogs** ([discogs.py](src/download/discogs.py), [discogs_dump.py](src/download/discogs_dump.py)):
  the same songs looked up on Discogs, to compare credit coverage
  (`credit_coverage()` in [exploration.py](src/exploration.py)).
- **Cleaning** ([clean.py](src/clean.py)): recordings of the same song (country
  releases, mono/single versions, Part 1/2, credit variants) are merged, combining their
  genres, credits and countries. Genius non-song pages, non-English lyrics and version
  pages (live, remaster, ...) are dropped. Each genre's MusicBrainz votes (tag counts,
  read from the checkpoints) are summed over the merged recordings (`genre_votes`).
  Every label is reduced to a core genre, the genre it is a kind of (`pop rock` -> `rock`,
  `proto-punk` -> `punk`, `west coast hip hop` -> `hip hop`, `rhythm & blues` -> `r&b`),
  and the votes are summed per core genre (`core_votes`). The primary `genre` is the
  most-voted core genre; ties are drawn at random, seeded per song. Instruments are
  grouped into families ([instruments.py](src/instruments.py)) as `instrument_families`.
- **Matching** ([combine.py](src/combine.py)): exact title + artist first, then looser
  rules, then fuzzy titles within the same artist. The core-genre votes are summed per
  coarse class: `genre_coarse` is the most-voted of rock, pop, rb, country, folk, blues,
  jazz, reggae, electronic, rap and other; `mb_label` the most-voted of the Genius tags
  (pop, rock, rb, country, rap) only, the genre used by the lyrics network and the
  classifier (rap dropped: too few songs). `label` is the Genius tag where there is one,
  and `labels_agree` marks songs where both sources agree.

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

The network analysis also drops filler words: vocables ("ooh", "la", "yeah") and what
spaCy leaves of contractions ("gonna" -> gon + na). Set `REMOVE_FILLER_WORDS = False` in
[src/config.py](src/config.py) to keep them; the cache is not changed. The classifier
and the lyrics network can leave out extra words with `excluded_words`.

## Environment note

If numpy crashes on Windows with error `0xc06d007f` (MKL with llvm-openmp in conda),
`src/__init__.py` already sets `MKL_THREADING_LAYER=TBB`, which avoids it. A lasting
fix is `conda install "libblas=*=*openblas"` in the environment.
