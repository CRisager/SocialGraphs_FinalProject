"""
Quick test: how well can a song's genre be predicted from its lyrics?

Compares three predictors on several label sets:
  guess   random guessing in proportion to how common each genre is
  counts  hand-made features: length, vocabulary richness, repeated lines, ...
  words   TF-IDF word weights + logistic regression (learns which words mark a genre)

All of an artist's songs are kept on the same side of each train/test split, so the
model cannot score by recognising an artist. Scores are macro-F1 (1.0 = perfect; every
genre counts equally, so always answering "rock" does not pay off).

Needs songs_<years>_lyrics.csv and genius_<years>_clean.csv (see dataCleaner/).

Usage:
    pip install scikit-learn
    python classification/classify_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataCleaner"))
from clean import YEARS  # noqa: E402  (the year range is set in clean.py)
SEED = 0


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


def test(name, songs, label_col):
    """Cross-validated macro-F1 per model; returns the word model's confusion matrix."""
    songs = songs.dropna(subset=["lyrics", label_col])
    X, y, artists = songs.lyrics.to_numpy(), songs[label_col].to_numpy(), songs.artist.to_numpy()
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    scores = {}
    for model_name, model in models().items():
        pred = cross_val_predict(model, X, y, groups=artists, cv=folds)
        scores[model_name] = f1_score(y, pred, average="macro")
    labels = sorted(set(y))
    print(f"{name:34s} {len(y):5d} songs   " +
          "   ".join(f"{m} {s:.2f}" for m, s in scores.items()))
    return pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels)


def main():
    matched = pd.read_csv(ROOT / f"songs_{YEARS}_lyrics.csv")
    genius = pd.read_csv(ROOT / f"genius_{YEARS}_clean.csv")
    genius = genius[genius.tag != "rap"]  # only 18 songs in 1970; hip hop starts ~1979

    print("MusicBrainz songs with lyrics")
    test("  Genius tag", matched, "genius_tag")
    test("  MusicBrainz label", matched, "mb_label")
    test("  only where both agree", matched[matched.labels_agree == True], "label")  # noqa: E712

    print(f"\nAll Genius songs from {YEARS}")
    with_pop = test("  Genius tag", genius, "tag")
    without_pop = test("  Genius tag, without pop", genius[genius.tag != "pop"], "tag")

    print("\nWord model on all Genius songs (rows = true genre, columns = predicted)")
    print(with_pop.to_string())
    print("\nSame without pop")
    print(without_pop.to_string())


if __name__ == "__main__":
    main()
