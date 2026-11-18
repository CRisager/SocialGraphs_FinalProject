import argparse
import csv
import unicodedata
from pathlib import Path


FIELD_SIZE_LIMIT = 128 * 1024 * 1024


def normalize(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def is_1970(value):
    if value is None:
        return False
    text = str(value).strip()
    return text == "1970" or text.startswith("1970.")


def load_1970_songs(metadata_path):
    songs = set()
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            if not is_1970(row.get("year")):
                continue
            title = normalize(row.get("song"))
            artist = normalize(row.get("artist"))
            if title and artist:
                songs.add((title, artist))
    return songs


def filter_lyrics(input_path, metadata_path, output_path):
    songs_1970 = load_1970_songs(metadata_path)
    csv.field_size_limit(FIELD_SIZE_LIMIT)

    written = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"title", "artist", "year", "lyrics"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns in {input_path}: {', '.join(sorted(missing))}"
            )

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                key = (normalize(row["title"]), normalize(row["artist"]))
                if is_1970(row["year"]) and key in songs_1970:
                    writer.writerow(row)
                    written += 1

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Filter the song lyrics CSV to lyrics matching songs released in 1970."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/song_lyrics.csv"),
        help="Source lyrics CSV.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("songs_1970.csv"),
        help="CSV containing the 1970 songs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/song_lyrics_1970.csv"),
        help="Destination CSV.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = filter_lyrics(args.input, args.metadata, args.output)
    print(f"Wrote {written} matching 1970 songs to {args.output}")


if __name__ == "__main__":
    main()
