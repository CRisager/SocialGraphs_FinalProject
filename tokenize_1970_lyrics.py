import argparse
import csv
import json
from collections import Counter
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


def remove_single_song_words(input_path, output_path=None, minimum_songs=2):
    if output_path is None:
        output_path = input_path

    csv.field_size_limit(FIELD_SIZE_LIMIT)
    document_frequency = Counter()

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        if "filtered_lyrics_tokens" not in fieldnames:
            raise ValueError(
                f"Missing required 'filtered_lyrics_tokens' column in {input_path}"
            )

        rows = list(reader)

    for row in rows:
        words = set(json.loads(row["filtered_lyrics_tokens"]))
        document_frequency.update(words)

    retained_words = {
        word
        for word, song_count in document_frequency.items()
        if song_count >= minimum_songs
    }

    fieldnames = [
        name
        for name in fieldnames
        if name not in {
            "cross_song_filtered_token_count",
            "cross_song_filtered_lyrics_tokens",
        }
    ]
    fieldnames.extend([
        "cross_song_filtered_token_count",
        "cross_song_filtered_lyrics_tokens",
    ])

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            words = [
                word
                for word in json.loads(row["filtered_lyrics_tokens"])
                if word in retained_words
            ]
            row["cross_song_filtered_token_count"] = len(words)
            row["cross_song_filtered_lyrics_tokens"] = json.dumps(words)
            writer.writerow(row)

    return len(retained_words)


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
    retained_words = remove_single_song_words(args.output, minimum_songs=2)
    print(f"Tokenized {written} songs to {args.output}")
    print(f"Retained {retained_words} words found in at least 2 songs")


if __name__ == "__main__":
    main()
