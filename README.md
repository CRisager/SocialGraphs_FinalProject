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

### Tokenizing the 1970 lyrics
Install `tiktoken`, then tokenize the filtered lyrics with the `o200k_base`
encoding:

```bash
pip install tiktoken
python3 tokenize_1970_lyrics.py
```

The output is written to `data/song_lyrics_1970_tokenized.csv` and includes
`token_count` plus a JSON list of token IDs in `lyrics_token_ids`.