"""
Clean the raw MusicBrainz (data/raw/songs_<year>.csv) and Genius
(data/raw/genius_songs_trimmed.csv) downloads for the years set in src/config.py.

  MusicBrainz - the same song often appears several times: released in several
                countries (one recording id each), as mono/single/instrumental
                versions, split into Part 1 / Part 2, or credited slightly differently
                ("Merle Haggard" vs "Merle Haggard and The Strangers"). These are merged
                into one row; genres, credits and release countries are combined.
  Genius      - non-song pages (tag "misc": poems, court rulings, speeches, ...) and
                non-English lyrics are dropped, version pages ("Mono", "Live",
                "2021 Remaster", "Alternate Mix", ...) are merged into the original,
                and [Verse]/[Chorus] headers are removed from the lyrics.

Genres (see the "Genres" section below for the details)
  `genre_votes`  every MusicBrainz genre of the song with its votes (user tag counts,
                 rebuilt from mb_<year>_checkpoint.json), summed over merged recordings
  `genres`       the same genres, most-voted first
  `core_votes`   the votes summed per core genre ("pop rock" counts for "rock")
  `genre`        the most-voted core genre; ties are drawn at random (seeded)
Instrument names are grouped into families (src/instruments.py) as
`instrument_families`.

The title/artist normalisation here is also used by combine.py to match the two.

Output (data/processed/; <years> is e.g. "1970-1980")
  songs_<years>_clean.csv    deduplicated MusicBrainz songs
  genius_<years>_clean.csv   cleaned Genius songs

Usage:
    python -m src.clean
"""

import json
import re
import unicodedata
import zlib
from collections import Counter

import numpy as np
import pandas as pd

from .config import (FIRST_YEAR, GENIUS_CLEAN, GENIUS_RAW, LAST_YEAR, PROCESSED_DIR,
                     SEED, SONGS_CLEAN, YEAR_RANGE, YEARS, musicbrainz_checkpoint,
                     musicbrainz_raw)
from .instruments import families, unknown

# Trailing version markers, matched on normalised (lowercase ascii) titles. Genius drops
# parentheses, so "Lola (mono single mix)" there is "Lola Mono Single Mix". Applied
# repeatedly, so "Teacher Mono UK Single Version" -> "teacher".
VERSION_SUFFIX = re.compile(r"\s+(?:" + "|".join([
    r"(?:\d{4}\s+)?remaster(?:ed)?(?:\s+\d{4})?",                 # 2021 remaster
    r"(?:day\s+\d+\s+)?demo(?:\s+version)?(?:\s+take\s+\d+)?",    # day 1 demo take 2
    r"session\s+outtakes\s+and\s+jams(?:\s+take\s+\d+)?",
    r"(?:remake\s+)?take\s+\d+",
    r"live(?:\s+at\s.*)?(?:\s+v\d)?",                             # live at the forum ...
    r"(?:(?:uk|us)\s+)?(?:[a-z0-9]+\s+)?version(?:\s+(?:\d+|one|two))?",  # single version
    r"(?:[a-z0-9]+\s+)?mix",                                      # alternate mix
    r"(?:single\s+)?edit",
    r"alternat(?:e|ive)(?:\s+(?:take|lyrics|vocals?))?",
    r"(?:mono|stereo)(?:\s+(?:uk|us))?",
    r"instrumental|reprise|outtake|acoustic",
]) + r")$")
PART_SUFFIX = re.compile(r"\s+(?:part|pt)\s*(?:\d+|i{1,3}|iv|one|two|three)(?:\s+(?:and\s+)?\d+)?$")
SECTION_HEADER = re.compile(r"^\s*[\[{][^\]}]*[\]}]\s*$", re.M)  # [Verse 1], {Instrumental}

# (title key, artist key) pairs: rows sharing any of them are the same song
MATCH_RULES = [
    (("title_key", "artist_key"), "title+artist"),
    (("title_core", "artist_key"), "title variant"),
    (("title_key", "artist_main"), "main artist"),
    (("title_core", "artist_main"), "main artist, title variant"),
]
KEY_COLS = ["title_key", "title_core", "artist_key", "artist_main"]
CREDIT_COLS = ["instruments", "producers", "musicians", "vocals", "labels"]
# songs_<year>.csv also has the crawler's own `genre` (its top-voted label); it is
# replaced by the vote-based core genre below


def ascii_words(s):
    """'Don’t Stop (Live) & Go' -> 'dont stop live and go'"""
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()
    s = s.replace("'", "").replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def title_key(t):
    return ascii_words(t).replace(" ", "") or t.lower()  # non-latin titles normalise to ""


def title_core(t):
    """Title without parentheses and version/part suffixes: 'Cool Strut, Part I' -> 'coolstrut'"""
    core, prev = ascii_words(re.sub(r"\(.*?\)|\[.*?\]", " ", t)), None
    while core != prev:
        prev, core = core, PART_SUFFIX.sub("", VERSION_SUFFIX.sub("", core))
    return core.replace(" ", "") or title_key(t)


def artist_key(a):
    return "".join(w for w in ascii_words(a).split() if w != "the") or a.lower()


def artist_main(a):
    """First credited artist: 'Ike & Tina Turner & The Ikettes' -> 'ike'"""
    first = re.split(r"\s+(?:feat\.?|featuring|ft\.?|with|&|and|vs\.?)\s+|\s*[,/]\s*", a,
                     flags=re.I)[0]
    return artist_key(first)


def add_keys(df, title, artist):
    return df.assign(title_key=df[title].map(title_key), title_core=df[title].map(title_core),
                     artist_key=df[artist].map(artist_key), artist_main=df[artist].map(artist_main))


def merge_groups(df, keys):
    """Label each row with a group id, where rows sharing a value in any of `keys`
    (lists of columns) end up in the same group, also through chains of rows."""
    gid = pd.Series(range(len(df)), index=df.index)
    while True:
        new = gid
        for cols in keys:
            new = new.groupby([df[c] for c in cols]).transform("min")
        if new.equals(gid):
            return gid
        gid = new


def join_unique(s):
    s = s.dropna().unique()
    return ";".join(s) if len(s) else None


def join_lists(s):
    """Merge ';'-joined lists: ['rock;pop', 'pop;soul', nan] -> 'rock;pop;soul'"""
    return join_unique(s.dropna().astype(str).str.split(";").explode())


assert join_lists(pd.Series(["rock;pop", "pop;soul", None])) == "rock;pop;soul"
assert join_lists(pd.Series([None, float("nan")])) is None


# ---------------------------------------------------------------------------------------
# Genres
#
# Votes. MusicBrainz genres are user tags; a tag's count is its up- minus down-votes. The
# crawler keeps the tags on MusicBrainz's official genre list with count > 0, from the
# recording itself if it has any, else pooled (summed) over its credited artists. The
# CSVs only keep the order, so the counts are rebuilt here from the checkpoints with the
# same rule, and summed over the recordings merged into one song.
#
# Core genres. A label is reduced to the genre it is a kind of. English genre compounds
# are head-final ("pop rock" is a kind of rock, "pop soul" a kind of soul), so:
#   1. the shortest ending of the label that is itself a genre label wins
#      ("proto-punk" -> "punk", "west coast hip hop" -> "hip hop", "new wave" -> "wave");
#      hyphens separate words, but one word is never split ("rocksteady" stays);
#   2. else the rightmost word that is a genre label by itself ("rock and roll" -> "rock");
#   3. else the label is a core genre itself ("beat music", "singer-songwriter").
# The vocabulary of genre labels is every label in the data. One fixed rule: "rhythm and
# blues" is always r&b, since the data otherwise spells it "r&b" and rule 1 would turn
# "british rhythm & blues" into blues.
#
# Primary genre. A song's votes are summed per core genre over all its labels and the
# most-voted core genre wins, so three rock subgenres with 1 vote each beat "pop" with 2.
# Ties are drawn at random rather than by list order, to avoid a systematic bias.
# ---------------------------------------------------------------------------------------

GENRE_SPLIT = re.compile(r"[\s\-]+")
RHYTHM_AND_BLUES = re.compile(r"\brhythm\s*(?:and|&|n)\s*blues\b")


def genre_words(label):
    return tuple(w for w in GENRE_SPLIT.split(str(label).strip().casefold()) if w)


def genre_vocabulary(genre_lists):
    """Word tuple -> label, for every label in a Series of ';'-joined genre lists."""
    return {genre_words(l): l.strip().casefold()
            for v in genre_lists.dropna() for l in v.split(";") if l.strip()}


def core_genre(label, vocab):
    """The core genre of one genre label (rules in the comment above)."""
    label = str(label).strip().casefold()
    if RHYTHM_AND_BLUES.search(label):
        return "r&b"
    words = genre_words(label)
    for i in range(len(words) - 1, 0, -1):  # proper endings, shortest first
        if words[i:] in vocab:
            # the ending can itself reduce further ("classic rock and roll" -> rock)
            return core_genre(vocab[words[i:]], vocab)
    for w in reversed(words[:-1]):
        if (w,) in vocab:
            return vocab[(w,)]
    return label


_vocab = genre_vocabulary(pd.Series(["rock;punk;pop;r&b;hip hop;wave;blues"]))
for _label, _core in [("pop rock", "rock"), ("proto-punk", "punk"), ("rock", "rock"),
                      ("west coast hip hop", "hip hop"), ("hip hop", "hip hop"),
                      ("new wave", "wave"), ("rock and roll", "rock"),
                      ("british rhythm & blues", "r&b"), ("beat music", "beat music"),
                      ("rocksteady", "rocksteady"), ("steady-rock", "rock"),
                      ("contemporary r&b", "r&b")]:
    assert core_genre(_label, _vocab) == _core, (_label, core_genre(_label, _vocab))


def pick_top(votes, key):
    """The most-voted item of a {item: votes} dict (None if empty). Ties are drawn at
    random with a generator seeded by SEED and `key` (the song id), so the draw is the
    same on every run and does not depend on the order of the songs."""
    if not votes:
        return None
    best = max(votes.values())
    tied = sorted(k for k, v in votes.items() if v == best)
    if len(tied) == 1:
        return tied[0]
    rng = np.random.default_rng([SEED, zlib.crc32(str(key).encode())])
    return tied[rng.integers(len(tied))]


assert pick_top({"rock": 3, "pop": 1}, "x") == "rock"
assert pick_top({}, "x") is None
assert pick_top({"rock": 2, "pop": 2}, "x") == pick_top({"pop": 2, "rock": 2}, "x")
assert {pick_top({"rock": 1, "pop": 1}, i) for i in range(50)} == {"rock", "pop"}


def votes_str(votes):
    """{'rock': 10, 'mod': 3} -> 'rock:10;mod:3', most-voted first (stable for ties)."""
    items = sorted(votes.items(), key=lambda kv: -kv[1])
    return ";".join(f"{k}:{v:g}" for k, v in items) or None


def parse_votes(s):
    """'rock:10;mod:3' -> {'rock': 10.0, 'mod': 3.0}; empty dict for a missing value."""
    if not isinstance(s, str) or not s:
        return {}
    return {k: float(v) for k, v in (item.rsplit(":", 1) for item in s.split(";"))}


assert parse_votes(votes_str({"rock": 10, "r&b": 3})) == {"rock": 10, "r&b": 3}


def load_tag_votes(years=YEAR_RANGE):
    """Recordings and artists (with their tag counts) from the crawler checkpoints."""
    recordings, artists = {}, {}
    for y in years:
        path = musicbrainz_checkpoint(y)
        if path.exists():
            cp = json.loads(path.read_text(encoding="utf-8"))
            recordings.update(cp.get("recordings", {}))
            artists.update(cp.get("artists", {}))
    return recordings, artists


def genre_votes(recording_id, genres, recordings, artists):
    """{genre: votes} of one recording's ';'-joined genre list, rebuilt the way the
    crawler chose the genres (own tags if any, else the artists' summed). Falls back to
    1 vote per genre when the checkpoint lacks the recording, keeping the list order."""
    if not isinstance(genres, str) or not genres:
        return {}
    names = genres.split(";")
    keep = set(names)

    def tally(tags):
        votes = {}
        for name, count in tags:
            if name in keep and count > 0:
                votes[name] = votes.get(name, 0) + count
        return votes

    rec = recordings.get(recording_id)
    if rec is not None:
        artist_ids, tags = rec[2], rec[3]
        votes = tally(tags) or tally([t for a in artist_ids for t in artists.get(a, [])])
        if set(votes) == keep:
            return {n: votes[n] for n in names}
    return dict.fromkeys(names, 1)


def clean_musicbrainz(mb):
    mb = add_keys(mb, "song", "artist")
    mb["group"] = merge_groups(mb, [cols for cols, _ in MATCH_RULES])
    recordings, artists = load_tag_votes()
    mb["votes"] = [genre_votes(r, g, recordings, artists) for r, g in zip(mb.id, mb.genres)]
    fallback = sum(isinstance(g, str) and r not in recordings for r, g in zip(mb.id, mb.genres))
    if fallback:
        print(f"Note: {fallback} recordings are not in a checkpoint; their genres get "
              f"1 vote each")
    # representative row: the shortest title, e.g. "Lola" over "Lola (mono single mix)"
    # and "Part 1" over "Part 2"
    songs = (mb.assign(n=mb.title_key.str.len()).sort_values(["n", "song"])
             .drop_duplicates("group").set_index("group").sort_index())
    grouped = mb.groupby("group")
    songs["countries"] = grouped.country.agg(join_unique)
    # the votes of a song are the sum over its recordings (a Counter sum keeps the
    # first-seen order, so equal votes stay in list order)
    votes = (grouped.votes.agg(lambda vs: sum((Counter(v) for v in vs), Counter()))
             .reindex(songs.index))
    songs["genre_votes"] = votes.map(votes_str)
    songs["genres"] = votes.map(lambda v: ";".join(sorted(v, key=lambda k: -v[k])) or None)
    vocab = genre_vocabulary(mb.genres)
    cores = {label: core_genre(label, vocab) for label in vocab.values()}
    core_votes = votes.map(lambda v: sum((Counter({cores[g.casefold()]: n})
                                          for g, n in v.items()), Counter()))
    songs["core_votes"] = core_votes.map(votes_str)
    songs["genre"] = [pick_top(v, i) for v, i in zip(core_votes, songs.id)]
    for col in CREDIT_COLS:  # absent in songs_<year>.csv written before the crawler had them
        if col in mb:
            songs[col] = grouped[col].agg(join_lists)
    if "instruments" in songs:
        songs["instrument_families"] = songs.instruments.map(families)
        missing = unknown(songs.instruments)
        if missing:
            print(f"Note: instruments not in src/instruments.py (counted as 'other'): "
                  f"{', '.join(missing)}")
    return songs.reset_index(drop=True).drop(columns=["n", "votes"] + KEY_COLS)


def clean_genius(g):
    g = g[(g.tag != "misc") & (g.language.eq("en") | g.language.isna())]
    g = g.assign(lyrics=g.lyrics.str.replace(SECTION_HEADER, "", regex=True)
                         .str.replace(r"\n\s*\n", "\n\n", regex=True).str.strip())
    g = add_keys(g[g.lyrics != ""], "title", "artist")
    g["group"] = merge_groups(g, [cols for cols, _ in MATCH_RULES]
                              + [["artist_key", "lyrics"], ["artist_main", "lyrics"]])
    # keep the original over its "Live" / "Alternate Mix" / ... pages
    return (g.assign(n=g.title_key.str.len()).sort_values("n", kind="stable")
            .drop_duplicates("group").sort_index().drop(columns=["n", "group"] + KEY_COLS))


def run():
    """Clean both raw datasets, write them to data/processed/ and return them."""
    mb_raw = pd.concat([pd.read_csv(musicbrainz_raw(y)) for y in YEAR_RANGE],
                       ignore_index=True)
    genius_raw = pd.read_csv(GENIUS_RAW)
    genius_raw = genius_raw[genius_raw.year.between(FIRST_YEAR, LAST_YEAR)]
    if genius_raw.year.nunique() < len(YEAR_RANGE):
        raise SystemExit(f"{GENIUS_RAW.name} lacks some of the years {YEARS}; "
                         f"re-run python -m src.download.genius")

    songs = clean_musicbrainz(mb_raw)
    print(f"MusicBrainz: {len(mb_raw)} recordings -> {len(songs)} songs")
    genius = clean_genius(genius_raw)
    misc = genius_raw.tag == "misc"
    foreign = ~misc & genius_raw.language.notna() & genius_raw.language.ne("en")
    print(f"Genius: {len(genius_raw)} pages -> {len(genius)} songs (dropped {misc.sum()} "
          f"non-song 'misc' pages, {foreign.sum()} non-English, "
          f"{len(genius_raw) - misc.sum() - foreign.sum() - len(genius)} duplicates/versions)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    songs.to_csv(SONGS_CLEAN, index=False)
    genius.to_csv(GENIUS_CLEAN, index=False)
    print(f"Wrote {SONGS_CLEAN.name}, {GENIUS_CLEAN.name}")
    return songs, genius


if __name__ == "__main__":
    run()
