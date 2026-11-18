"""
How well can a song's genre be predicted from its lyrics?

Compares three predictors on several label sets:
  guess   random guessing in proportion to how common each genre is
  counts  hand-made features: length, vocabulary richness, repeated lines, ...
  words   TF-IDF word weights + logistic regression (learns which words mark a genre);
          spaCy tokens as in the lyric graph (src/text.py)

Genre is the MusicBrainz 5-label scheme mb_label (rap dropped: too few songs), with the
Genius tag and the songs where both sources agree as checks, and all Genius songs of
the period (with and without pop) as a larger sample.

All of an artist's songs (credits like "X and The Y" count as X) are kept on the same
side of each train/test split, so the model cannot score by recognising an artist.
Scores are macro-F1 (1.0 = perfect; every genre counts equally, so always answering
"rock" does not pay off).

Usage:
    python -m src.classification
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .clean import artist_main
from .config import YEARS
from .data import load_data
from .text import lyrics_vectorizer, tokens_for

SEED = 0  # kept separate from config.SEED so scores match earlier runs


def count_features(texts):
    """Per song: log words, unique-word share, log lines, repeated-line share,
    mean word length, words per line."""
    rows = []
    for text in texts:
        words = text.lower().split()
        lines = [line.strip().lower() for line in text.split("\n") if line.strip()]
        rows.append([np.log(len(words)), len(set(words)) / len(words),
                     np.log(len(lines)), 1 - len(set(lines)) / len(lines),
                     np.mean([len(w) for w in words]), len(words) / len(lines)])
    return np.array(rows)


assert count_features(["la la\nla la\nhey"])[0][3] == 1 - 2 / 3  # one of three lines repeats


def _raw_lyrics(X):
    return X["lyrics"]


def _tokens(X):
    return X["tokens"]


def models():
    return {
        "guess": DummyClassifier(strategy="stratified", random_state=SEED),
        "counts": make_pipeline(FunctionTransformer(_raw_lyrics),
                                FunctionTransformer(count_features), StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "words": make_pipeline(FunctionTransformer(_tokens), lyrics_vectorizer(min_df=2),
                               LogisticRegression(max_iter=2000, class_weight="balanced")),
    }


def test(name, songs, label_col, verbose=True):
    """Cross-validated macro-F1 per model. Returns (scores as a pd.Series, the word
    model's confusion matrix)."""
    songs = songs.dropna(subset=["lyrics", label_col])
    X = pd.DataFrame({"lyrics": songs.lyrics, "tokens": tokens_for(songs.lyrics, verbose)})
    y, artists = songs[label_col].to_numpy(), songs.artist.map(artist_main).to_numpy()
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    scores = {}
    for model_name, model in models().items():
        pred = cross_val_predict(model, X, y, groups=artists, cv=folds)
        scores[model_name] = f1_score(y, pred, average="macro")
    labels = sorted(set(y))
    if verbose:
        print(f"{name:42s} {len(y):5d} songs   " +
              "   ".join(f"{m} {s:.2f}" for m, s in scores.items()))
    confusion = pd.DataFrame(confusion_matrix(y, pred, labels=labels),
                             index=labels, columns=labels)
    return pd.Series({"songs": len(y), **scores}, name=name), confusion


def run_all(songs, genius, verbose=True):
    """Every label set compared in the report. Returns (macro-F1 table, confusion
    matrices of the word model: main label set, Genius songs with and without pop)."""
    matched = songs[songs.mb_label != "rap"]
    genius = genius[genius.tag != "rap"]  # only ~400 songs; hip hop starts ~1979
    sets = [
        ("MusicBrainz songs: MusicBrainz label", matched, "mb_label"),
        ("MusicBrainz songs: Genius tag", matched[matched.genius_tag != "rap"], "genius_tag"),
        ("MusicBrainz songs: both labels agree", matched[matched.labels_agree == True],  # noqa: E712
         "label"),
        (f"Genius {YEARS}: tag", genius, "tag"),
        (f"Genius {YEARS}: tag, without pop", genius[genius.tag != "pop"], "tag"),
    ]
    scores, confusions = [], {}
    for name, data, label_col in sets:
        score, confusion = test(name, data, label_col, verbose)
        scores.append(score)
        confusions[name] = confusion
    table = pd.DataFrame(scores).astype({"songs": int})
    return table, {"MusicBrainz label": confusions[sets[0][0]],
                   "Genius, with pop": confusions[sets[3][0]],
                   "Genius, without pop": confusions[sets[4][0]]}


def main():
    songs, genius = load_data()
    _, confusions = run_all(songs, genius)
    for name, confusion in confusions.items():
        print(f"\nWord model, {name} (rows = true genre, columns = predicted)")
        print(confusion.to_string())


if __name__ == "__main__":
    main()
