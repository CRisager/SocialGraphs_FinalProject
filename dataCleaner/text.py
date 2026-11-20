"""
Word-level text preprocessing shared by the lyric graph and the genre classifier.

Lyrics are lowercased and split into whole words (letters, with one inner apostrophe
kept: "don't", "lovin"), not sub-word GPT tokens, so TF-IDF vocabularies and
distinctive-word lists are made of real words. English stop words are removed,
including their contracted forms ("i'm", "don't"), which the scikit-learn list lacks,
and single letters (the "n" of "rock n roll", spelled-out words like "s-t-o-p").
Dialect words such as "ain't" and "gonna" are kept: they may mark a genre.
"""

from string import ascii_lowercase
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

TOKEN_PATTERN = r"\b[a-z]+(?:'[a-z]+)?\b"
CONTRACTIONS = """i'm i've i'll i'd you're you've you'll you'd he's he'll he'd she's she'll
she'd it's it'll we're we've we'll we'd they're they've they'll they'd that's there's
what's where's who's let's don't doesn't didn't isn't aren't wasn't weren't can't couldn't
won't wouldn't shouldn't haven't hasn't hadn't""".split()
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | set(CONTRACTIONS) | set(ascii_lowercase))


def normalise(text):
    return text.lower().replace("’", "'")  # Genius mixes ’ and '


def remove_words(text, excluded_words=()):
    """Return text with whole-word matches removed; the source CSV is unchanged."""
    text = normalise(text)
    words = {normalise(word).strip("'") for word in excluded_words if word}
    for word in sorted(words, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(word)}\b", " ", text)
    return text


def lyrics_vectorizer(excluded_words=(), **kwargs):
    excluded_words = {normalise(word).strip("'") for word in excluded_words if word}
    return TfidfVectorizer(preprocessor=normalise, token_pattern=TOKEN_PATTERN,
                           stop_words=sorted(set(STOP_WORDS) | excluded_words),
                           sublinear_tf=True, **kwargs)


assert list(lyrics_vectorizer().build_analyzer()("I’m lovin' you, I ain't 2 GONNA s-t-o-p")) \
    == ["lovin", "ain't", "gonna"]
