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