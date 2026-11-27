"""
What the communities contain (Step 6D): words, sentiment and instruments.

Words. Each community with at least MIN_SIZE songs becomes one document, and the
communities of one graph are compared with TF-IDF:
    tf(w, c)  = number of songs in community c whose lyrics use w (once per song)
    idf(w)    = log(n_communities / number of communities that use w)
Counting songs rather than word occurrences matters here: plain TF-IDF on the merged
lyrics was dominated by names and by words one song repeats many times ("chh",
"suzanne", "ding dong"), which say nothing about a community. A word used in every
community gets idf = 0, since it cannot tell them apart. Only words used in at least
MIN_SONGS songs of the graph are considered, so a single song cannot define a theme.

Sentiment. VADER (Hutto & Gilbert 2014) is a lexicon for short texts: on a whole song
its compound score sums the valence of every word and quickly saturates at -1 or +1.
So each lyric line is scored on its own and the song's sentiment is the mean compound
score of its non-empty lines (-1 = negative, +1 = positive). VADER needs the raw text
(capitals, punctuation and negations like "not" change the score), not the tokens.
Whether communities differ more in sentiment than random groups would is tested by
shuffling the song scores over the songs (keeping the group sizes) and comparing the
spread of the group means (the between-group sum of squares).
"""

from collections import Counter

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .nulls import empirical_p

MIN_SIZE = 10    # communities smaller than this are not described
MIN_SONGS = 5    # a word must be used by this many songs of the graph


def community_tokens(member, tokens, min_size=MIN_SIZE):
    """{community: list of token lists} for the songs with lyrics, communities with at
    least min_size songs (counted on all songs, with or without lyrics)."""
    member = pd.Series(member)
    sizes = member.value_counts()
    keep = sorted(sizes[sizes >= min_size].index)
    tok = tokens.reindex(member.index).dropna()
    return {c: list(tok[member.reindex(tok.index) == c]) for c in keep}


def community_tfidf(groups, min_songs=MIN_SONGS):
    """DataFrame (rows = communities, columns = words) of song-frequency TF-IDF."""
    sets = {c: [set(t) for t in songs] for c, songs in groups.items()}
    song_df = Counter(w for ss in sets.values() for s in ss for w in s)
    vocab = sorted(w for w, n in song_df.items() if n >= min_songs)
    tf = pd.DataFrame({c: Counter(w for s in ss for w in s) for c, ss in sets.items()}).T
    tf = tf.reindex(columns=vocab).fillna(0)
    idf = np.log(len(tf) / (tf > 0).sum(axis=0))
    return tf * idf


def top_words(scores, n=10):
    """{community: 'w1 w2 ...'} with the n highest-scoring words."""
    return {c: " ".join(row.nlargest(n).index) for c, row in scores.iterrows()}


_VADER = SentimentIntensityAnalyzer()


def song_sentiment(lyrics):
    """Mean VADER compound score over the non-empty lines of a song."""
    lines = [l.strip() for l in str(lyrics).splitlines() if l.strip()]
    if not lines:
        return np.nan
    return float(np.mean([_VADER.polarity_scores(l)["compound"] for l in lines]))


def between_ss(values, groups):
    """Between-group sum of squares: sum over groups of size * (group mean - mean)^2."""
    df = pd.DataFrame({"v": values, "g": groups})
    m = df.v.mean()
    g = df.groupby("g").v.agg(["size", "mean"])
    return float((g["size"] * (g["mean"] - m) ** 2).sum())


def permutation_groups(values, groups, n, seed):
    """(real between-group SS, share of total SS, p-value from n shuffles of values)."""
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    rng = np.random.default_rng(seed)
    real = between_ss(values, groups)
    null = np.array([between_ss(rng.permutation(values), groups) for _ in range(n)])
    total = float(((values - values.mean()) ** 2).sum())
    return real, real / total, empirical_p(real, null)


def family_profile(member, families, min_size=MIN_SIZE):
    """Community x instrument family: % of the community's songs with each family.
    `families` maps song -> 'fam1;fam2;...' (songs without data left out)."""
    member = pd.Series(member)
    fam = families.reindex(member.index).dropna().str.get_dummies(sep=";")
    sizes = member.value_counts()
    keep = sorted(sizes[sizes >= min_size].index)
    prof = fam.groupby(member.reindex(fam.index)).mean().mul(100)
    return prof.reindex(keep)


# toy checks
_g = {0: [["sun", "love"], ["sun", "love"], ["love"]], 1: [["rain", "love"], ["rain"]]}
_s = community_tfidf(_g, min_songs=1)
assert _s.loc[0, "sun"] > 0 and _s.loc[0, "love"] == 0 and _s.loc[1, "rain"] > _s.loc[1, "sun"]
assert song_sentiment("I love you\n\nwonderful day") > 0.5 and song_sentiment("I hate this") < 0
assert np.isnan(song_sentiment(""))
assert between_ss([1, 1, 3, 3], [0, 0, 1, 1]) == 4.0
