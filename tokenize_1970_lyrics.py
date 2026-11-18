import argparse
import csv
import json
from pathlib import Path

import spacy


FIELD_SIZE_LIMIT = 128 * 1024 * 1024
STRUCTURAL_TOKENS = {"verse", "chorus", "bridge", "intro", "outro", "hook", "refrain"}


def tokenize(input_path, output_path, model_name="en_core_web_sm"):
    nlp = spacy.load(model_name)
    csv.field_size_limit(FIELD_SIZE_LIMIT)
    written = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if "lyrics" not in (reader.fieldnames or []):
            raise ValueError(f"Missing required 'lyrics' column in {input_path}")

        fieldnames = list(reader.fieldnames or [])
        fieldnames.extend([
            "token_count",
            "lyrics_tokens",
            "filtered_token_count",
            "filtered_lyrics_tokens",
        ])

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                lowercase_lyrics = (row.get("lyrics") or "").lower()
                doc = nlp(lowercase_lyrics)
                tokens = [token.text for token in doc]
                filtered_tokens = [
                    token.text
                    for token in doc
                    if token.is_alpha
                    and not token.is_stop
                    and token.text not in STRUCTURAL_TOKENS
                ]
                row["token_count"] = len(tokens)
                row["lyrics_tokens"] = json.dumps(tokens)
                row["filtered_token_count"] = len(filtered_tokens)
                row["filtered_lyrics_tokens"] = json.dumps(filtered_tokens)
                writer.writerow(row)
                written += 1

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize a lyrics CSV with spaCy."
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
        "--model",
        default="en_core_web_sm",
        help="spaCy model name.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = tokenize(args.input, args.output, args.model)
    print(f"Tokenized {written} songs to {args.output}")


if __name__ == "__main__":
    main()
