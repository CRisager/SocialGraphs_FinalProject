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

## Tokenizing the 1970 lyrics

Install spaCy and download the English model, then tokenize the filtered lyrics:

```bash
pip install spacy
python -m spacy download en_core_web_sm
python3 tokenize_1970_lyrics.py
```

The output is written to `data/song_lyrics_1970_tokenized.csv`.

### Tokenization method

The project uses spaCy's `en_core_web_sm` English pipeline for linguistic
tokenization. The pipeline produces whole words such as `whisper` and
`speaking`, although spaCy's rule-based tokenizer may still split contractions
or unusual punctuation according to its tokenization rules.

Before spaCy processes each song, the complete lyrics string is converted to
lowercase with Python's `str.lower()`. This makes capitalization consistent in
the resulting token lists.

### Filtering applied to lyrics tokens

The filtered token list keeps a token only when all of the following are true:

1. `token.is_alpha` is true, so punctuation, symbols, and other non-alphabetic
   tokens—including numbers and alphanumeric tokens—are removed.
2. `token.is_stop` is false. Stopwords come from the standard English stopword
   list bundled with spaCy (`en_core_web_sm`), and are determined by spaCy rather
   than a project-specific stopword list.
3. The lowercased token text is not one of the project-specific structural
   section markers: `verse`, `chorus`, `bridge`, `intro`, `outro`, `hook`, or
   `refrain`.

Punctuation is not a separate list or regular-expression substitution; it is
excluded by the `token.is_alpha` check. Likewise, stopwords are not decoded or
hard-coded in this script: they are identified using the stopword list supplied
by spaCy's English model.

### Tokenized CSV fields

The output CSV retains all original lyrics fields and adds:

- `token_count`: number of all spaCy tokens before filtering.
- `lyrics_tokens`: JSON list of all spaCy token strings.
- `filtered_token_count`: number of tokens remaining after the filtering
  described above.
- `filtered_lyrics_tokens`: JSON list of the remaining whole-word token strings.

The original, unfiltered token fields remain available so that filtering choices
can be inspected or changed without retokenizing from the source lyrics.

### Counting the most frequent words

The notebook aggregates `filtered_lyrics_tokens` across the 303 rows in
`data/song_lyrics_1970_tokenized.csv` using Python's `collections.Counter`.
Counts therefore include all selected 1970 songs in the output dataset, not only
`Let It Be`. The example-song cell counts only the first `Let It Be` row.