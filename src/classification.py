"""
Quick test: how well can a song's genre be predicted from its lyrics?

Compares three predictors on several label sets:
  guess   random guessing in proportion to how common each genre is
  counts  hand-made features: length, vocabulary richness, repeated lines, ...
  words   TF-IDF word weights + logistic regression (learns which words mark a genre)

All of an artist's songs are kept on the same side of each train/test split, so the
model cannot score by recognising an artist. Scores are macro-F1 (1.0 = perfect; every
genre counts equally, so always answering "rock" does not pay off).

Usage:
    python -m src.classification
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import YEARS
from .data import load_data

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


def models():
    return {
        "guess": DummyClassifier(strategy="stratified", random_state=SEED),
        "counts": make_pipeline(FunctionTransformer(count_features), StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "words": make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2),
                               LogisticRegression(max_iter=2000, class_weight="balanced")),
    }


def test(name, songs, label_col, verbose=True):
    """Cross-validated macro-F1 per model. Returns (scores as a pd.Series, the word
    model's confusion matrix)."""
    songs = songs.dropna(subset=["lyrics", label_col])
    X, y, artists = songs.lyrics.to_numpy(), songs[label_col].to_numpy(), songs.artist.to_numpy()
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    scores = {}
    for model_name, model in models().items():
        pred = cross_val_predict(model, X, y, groups=artists, cv=folds)
        scores[model_name] = f1_score(y, pred, average="macro")
    labels = sorted(set(y))
    if verbose:
        print(f"{name:34s} {len(y):5d} songs   " +
              "   ".join(f"{m} {s:.2f}" for m, s in scores.items()))
    confusion = pd.DataFrame(confusion_matrix(y, pred, labels=labels),
                             index=labels, columns=labels)
    return pd.Series({"songs": len(y), **scores}, name=name), confusion


def run_all(songs, genius, verbose=True):
    """Every label set compared in the report. Returns (macro-F1 table, confusion
    matrices of the word model on the Genius songs, with and without pop)."""
    genius = genius[genius.tag != "rap"]  # only ~400 songs; hip hop starts ~1979
    agree = songs[songs.labels_agree == True]  # noqa: E712
    sets = [
        ("MusicBrainz songs: Genius tag", songs, "genius_tag"),
        ("MusicBrainz songs: MusicBrainz label", songs, "mb_label"),
        ("MusicBrainz songs: both labels agree", agree, "label"),
        (f"Genius {YEARS}: tag", genius, "tag"),
        (f"Genius {YEARS}: tag, without pop", genius[genius.tag != "pop"], "tag"),
    ]
    scores, confusions = [], {}
    for name, data, label_col in sets:
        score, confusion = test(name, data, label_col, verbose)
        scores.append(score)
        confusions[name] = confusion
    table = pd.DataFrame(scores).astype({"songs": int})
    return table, {"with pop": confusions[sets[3][0]],
                   "without pop": confusions[sets[4][0]]}


def main():
    songs, genius = load_data()
    _, confusions = run_all(songs, genius)
    for name, confusion in confusions.items():
        print(f"\nWord model on Genius songs {name} (rows = true genre, columns = predicted)")
        print(confusion.to_string())


if __name__ == "__main__":
    main()
