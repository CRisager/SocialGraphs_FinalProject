import os
import shutil
import kagglehub
import pandas as pd

handle = "carlosgdcj/genius-song-lyrics-with-language-information"

# Only load specific columns into your DataFrame
columns_to_keep = ["title", "artist", "year", "tag", "language"]

# Downloads only song_lyrics.csv into the kagglehub cache and returns its path
csv_path = kagglehub.dataset_download(handle, path="song_lyrics.csv")

df = pd.read_csv(csv_path, usecols=columns_to_keep)

print(df.head())

df.to_csv("./genius_songs_trimmed.csv", index=False)

# Remove this dataset's cached copy
cache_root = os.environ.get("KAGGLEHUB_CACHE", os.path.expanduser("~/.cache/kagglehub"))
dataset_cache = os.path.join(cache_root, "datasets", *handle.split("/"))
shutil.rmtree(dataset_cache, ignore_errors=True)
print(f"Removed cache: {dataset_cache}")
