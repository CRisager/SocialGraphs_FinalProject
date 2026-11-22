"""
Lyric tokenization shared by the lyrics network, the genre classifier and the overview.

Method (from the token branch): each song's lyrics are lowercased with str.lower() and
tokenized with spaCy's English tokenizer (en_core_web_sm). A token is kept when
  1. token.is_alpha        - punctuation, numbers and alphanumeric tokens are dropped
  2. not token.is_stop     - spaCy's English stop-word list
  3. it is not a section marker: verse, chorus, bridge, intro, outro, hook, refrain
Words that occur in only one song are dropped by the TF-IDF (min_df=2), which is the
same as the token branch's cross-song filter.

Only the tokenizer runs (no tagger/parser/NER): is_alpha and is_stop are properties of
the word itself, so the tokens are identical to the full pipeline's, just much faster.
Tokens are cached in data/processed/tokens.csv (keyed by a hash of the lyrics), so
each song is tokenized only once.

Setup:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import hashlib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import PROCESSED_DIR

MODEL = "en_core_web_sm"
STRUCTURAL_TOKENS = {"verse", "chorus", "bridge", "intro", "outro", "hook", "refrain"}
TOKEN_CACHE = PROCESSED_DIR / "tokens.csv"

_nlp = None


def _tokenizer():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load(MODEL, exclude=["tok2vec", "tagger", "parser", "attribute_ruler",
                                          "lemmatizer", "ner", "senter"])
    return _nlp


def tokenize(texts, batch_size=500):
    """List of filtered tokens for each text."""
    return [[t.text for t in doc
             if t.is_alpha and not t.is_stop and t.text not in STRUCTURAL_TOKENS]
            for doc in _tokenizer().pipe((str(t).lower() for t in texts),
                                         batch_size=batch_size)]


def _key(text):
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()


def tokens_for(lyrics, verbose=True):
    """Filtered tokens for each lyrics string (a pd.Series), aligned with its index.
    Uses the cache; only new lyrics are tokenized."""
    keys = lyrics.map(_key)
    cache = (pd.read_csv(TOKEN_CACHE, keep_default_na=False).set_index("key")["tokens"]
             if TOKEN_CACHE.exists() else pd.Series(dtype=str))
    todo = lyrics[~keys.isin(cache.index)].drop_duplicates()
    if len(todo):
        if verbose:
            print(f"Tokenizing {len(todo)} lyrics with spaCy (cached afterwards)...")
        new = pd.Series([" ".join(t) for t in tokenize(todo)],
                        index=todo.map(_key).to_numpy())
        cache = pd.concat([cache, new])
        cache = cache[~cache.index.duplicated()]
        TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        cache.rename_axis("key").rename("tokens").to_csv(TOKEN_CACHE)
    return pd.Series([cache[k].split() for k in keys], index=lyrics.index)


# ---------------------------------------------------------------------------------------
# Filler words (used by src/dataset.py when config.REMOVE_FILLER_WORDS is True)
#
# Vocables ("ooh", "la la la", "yeah") carry no theme but are frequent enough to become
# a song's top TF-IDF words and to link unrelated songs. The tokenizer fragments are
# what spaCy leaves of contractions ("gonna" -> gon + na, "gotta" -> got + ta,
# "won't" -> wo, "ain't" -> ai; the "n't" part is not alphabetic and already dropped).
# "ya" is a spelling of "you", which is a stop word. Single letters ("rock 'n' roll")
# are dropped too. Real words that are part of a song's theme (baby, love, girl, yes,
# alright, wow) are kept.
#
# The token cache (tokens.csv) is never changed: the words are removed when the analysis
# loads its tokens, so turning REMOVE_FILLER_WORDS off restores the unfiltered tokens.
# ---------------------------------------------------------------------------------------

FILLER_WORDS = frozenset("""
    oh ohh ohhh ooh oooh oo ooo oooo ah ahh aah aw ow
    la lala na da dah doo dum dee ba bam bop sha shoo wah
    hey yeah yea yeh yay uh huh mmm mm hmm whoa woah woo whoo hoo ay ayy ho ha hah yo
""".split())
TOKENIZER_FRAGMENTS = frozenset("gon na ta wo ai ya nt ll ve".split())


def drop_fillers(tokens, words=FILLER_WORDS | TOKENIZER_FRAGMENTS, min_len=2):
    """Tokens without the given filler words and without tokens shorter than min_len."""
    return [t for t in tokens if len(t) >= min_len and t not in words]


assert drop_fillers(["oh", "yeah", "baby", "gon", "na", "love", "n", "rock"]) == ["baby", "love", "rock"]


def _identity(tokens):
    return tokens


def lyrics_vectorizer(**kwargs):
    """TF-IDF over pre-tokenized lyrics (lists of tokens from tokens_for)."""
    return TfidfVectorizer(analyzer=_identity, sublinear_tf=True, **kwargs)
