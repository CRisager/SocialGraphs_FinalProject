# SocialGraphs_FinalProject

### Downloading the datasets

Run ```python dataDownload/crawler-musicBrainz.py --years 1970 1980``` (one
```songs_<year>.csv``` per year, resumable) and
```python dataDownload/download-Genius.py``` (keeps Genius songs from
1970-1980). The year range used by everything after the download is set by
```FIRST_YEAR, LAST_YEAR``` in ```dataCleaner/clean.py```. Each song gets its
genres and the instruments credited on the recording (column
```instruments```, empty where MusicBrainz has no credits). The instrument
phase needs one request per release, about 5.5 hours for 1970-1980; add
```--no-instruments``` to skip it.

### Cleaning and combining

Run ```dataCleaner/clean.py``` to clean both datasets (duplicates, version
pages, non-song Genius pages), which writes
```songs_1970-1980_clean.csv``` and ```genius_1970-1980_clean.csv```. Then run
```dataCleaner/combine.py``` to attach Genius lyrics to the MusicBrainz songs,
which writes ```songs_1970-1980_lyrics.csv```.

During cleaning, primary genre labels are reduced to automatically discovered
single-word core genres. For compound labels, the rightmost recognized core
genre wins (for example, `pop rock` becomes `rock`). The original full labels
remain in `genres`.

### Interactive genre network

For a temporary interactive circular view of the genre network, run:

```bash
python genre_network_visualization.py
```

This writes `analysis/genre_network.html`. Songs are connected when they share
an overall genre or subgenre, node colors use the primary genre, and hovering
over a node shows its song, artist, and year while highlighting its edges.
When available, the script automatically uses Jonathan's
`songs_1970_lyrics.csv` output.
For large datasets, the default is 500 sampled edges per genre; change this
with `--max-edges-per-genre`.
