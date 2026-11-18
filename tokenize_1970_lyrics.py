import argparse
import csv
import json
from pathlib import Path

import tiktoken


FIELD_SIZE_LIMIT = 128 * 1024 * 1024


def tokenize(input_path, output_path, encoding_name="o200k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    csv.field_size_limit(FIELD_SIZE_LIMIT)
    written = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if "lyrics" not in (reader.fieldnames or []):
            raise ValueError(f"Missing required 'lyrics' column in {input_path}")

        fieldnames = list(reader.fieldnames or [])
        fieldnames.extend(["token_count", "lyrics_token_ids"])

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                token_ids = encoding.encode(row.get("lyrics") or "", disallowed_special=())
                row["token_count"] = len(token_ids)
                row["lyrics_token_ids"] = json.dumps(token_ids)
                writer.writerow(row)
                written += 1

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a lyrics CSV with tiktoken's o200k_base encoding."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/song_lyrics_1970.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/song_lyrics_1970_tokenized.csv"),
    )
    parser.add_argument(
        "--encoding",
        default="o200k_base",
        help="Tiktoken encoding name.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = tokenize(args.input, args.output, args.encoding)
    print(f"Tokenized {written} songs to {args.output}")


if __name__ == "__main__":
    main()
