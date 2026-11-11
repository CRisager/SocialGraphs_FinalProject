# SocialGraphs_FinalProject

### Downloading the datasets
Run ```Crawler-musicBrainz.py``` and ```download-Genius.py```. Do note the crawler may get timed out if run too often as to not break rules regarding rate limits. Additionally the Genius dataset downloader needs at least 9gb free space on the disk.

### Filtering the lyrics dataset
To keep only lyrics whose title and artist occur in `songs_1970.csv`, run:

```bash
python3 filter_song_lyrics.py
```

The matching rows are written to `data/song_lyrics_1970.csv`. The lyrics CSV is
processed as a stream, so the complete dataset does not need to fit in memory.

### Data processing and exploration

After both source datasets have been generated, run. The commands below assume
the CSV files are in the repository root; the notebook also detects `Data/`
and `data/` folders automatically:

```bash
python filter_song_lyrics.py
python data_exploration.py
```

Alternatively, open `main.ipynb` and execute its cells in order to inspect each
stage interactively.

The analysis writes to `analysis/`: cleaned song data, lyric TF-IDF and
genre features, exploratory plots, K-means labels, a shared-genre/artist
community graph (`song_similarity.gexf`), and community assignments.
Lyrics are lower-cased, section labels and punctuation are removed, and
matching uses normalized title/artist keys. The feature set includes primary
metadata, multi-label genres, and lyric unigrams/bigrams. Community detection
uses greedy modularity on a weighted song projection where songs are linked
when they share an artist or genre.