"""Filter the Genius lyrics dump to songs found in the MusicBrainz dataset."""

import argparse
import csv
import re
import unicodedata
from pathlib import Path


def normalise(value: str) -> str:
    """Create a stable key for matching titles and artists across datasets."""
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def song_key(title: str, artist: str) -> tuple[str, str]:
    return normalise(title), normalise(artist)


def filter_lyrics(songs_path: Path, lyrics_path: Path, output_path: Path) -> int:
    with songs_path.open(newline="", encoding="utf-8-sig") as songs_file:
        songs = csv.DictReader(songs_file)
        wanted = {
            song_key(row.get("song", ""), row.get("artist", ""))
            for row in songs
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with lyrics_path.open(newline="", encoding="utf-8-sig") as source:
        lyrics = csv.DictReader(source)
        fieldnames = lyrics.fieldnames or []
        required = {"title", "artist", "lyrics"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"{lyrics_path} is missing columns: {sorted(missing)}")
        with output_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for row in lyrics:
                if song_key(row.get("title", ""), row.get("artist", "")) in wanted:
                    writer.writerow(row)
                    written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", type=Path, default=Path("songs_1970.csv"))
    parser.add_argument("--lyrics", type=Path, default=Path("genius_songs_trimmed.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/song_lyrics_1970.csv")
    )
    args = parser.parse_args()
    count = filter_lyrics(args.songs, args.lyrics, args.output)
    print(f"Wrote {count} matching lyric rows to {args.output}")


if __name__ == "__main__":
    main()
