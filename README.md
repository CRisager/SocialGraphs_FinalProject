# SocialGraphs_FinalProject

### Downloading the datasets
Run ```python dataDownload/crawler-musicBrainz.py --years 1970 1980``` (one ```songs_<year>.csv``` per year, resumable) and ```python dataDownload/download-Genius.py``` (keeps Genius songs from 1970-1980). The year range used by everything after the download is set by ```FIRST_YEAR, LAST_YEAR``` in ```dataCleaner/clean.py```. Each song gets its genres and the instruments credited on the recording (column ```instruments```, empty where MusicBrainz has no credits). The instrument phase needs one request per release, about 5.5 hours for 1970-1980; add ```--no-instruments``` to skip it. Do note the crawler may get timed out if run too often as to not break rules regarding rate limits. Additionally the Genius dataset downloader needs at least 9gb free space on the disk.

### Cleaning and combining
Run ```dataCleaner/clean.py``` to clean both datasets (duplicates, version pages, non-song Genius pages), which writes ```songs_1970-1980_clean.csv``` and ```genius_1970-1980_clean.csv```. Then run ```dataCleaner/combine.py``` to attach Genius lyrics to the MusicBrainz songs, which writes ```songs_1970-1980_lyrics.csv```.