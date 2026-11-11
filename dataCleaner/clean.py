"""
Clean songs_<year>.csv (MusicBrainz) and genius_songs_trimmed.csv (Genius) for the years
FIRST_YEAR..LAST_YEAR set below.

  MusicBrainz - the same song often appears several times: released in several
                countries (one recording id each), as mono/single/instrumental
                versions, split into Part 1 / Part 2, or credited slightly differently
                ("Merle Haggard" vs "Merle Haggard and The Strangers"). These are merged
                into one row; genres, instruments and release countries are combined.
  Genius      - non-song pages (tag "misc": poems, court rulings, speeches, ...) and
                non-English lyrics are dropped, version pages ("Mono", "Live",
                "2021 Remaster", "Alternate Mix", ...) are merged into the original,
                and [Verse]/[Chorus] headers are removed from the lyrics.

The title/artist normalisation here is also used by combine.py to match the two.

Output (repo root; <years> is e.g. "1970-1980", or "1970" for one year)
  songs_<years>_clean.csv    deduplicated MusicBrainz songs
  genius_<years>_clean.csv   cleaned Genius songs

Usage:
    python dataCleaner/clean.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Years to use. Every script after this one reads the files named by YEARS.
FIRST_YEAR, LAST_YEAR = 1970, 1980
YEARS = str(FIRST_YEAR) if FIRST_YEAR == LAST_YEAR else f"{FIRST_YEAR}-{LAST_YEAR}"

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


def clean_musicbrainz(mb):
    mb = add_keys(mb, "song", "artist")
    mb["group"] = merge_groups(mb, [cols for cols, _ in MATCH_RULES])
    # representative row: the shortest title, e.g. "Lola" over "Lola (mono single mix)"
    # and "Part 1" over "Part 2"
    songs = (mb.assign(n=mb.title_key.str.len()).sort_values(["n", "song"])
             .drop_duplicates("group").set_index("group").sort_index())
    grouped = mb.groupby("group")
    songs["countries"] = grouped.country.agg(join_unique)
    songs["genres"] = grouped.genres.agg(join_lists)
    if "instruments" in mb:  # absent in songs_<year>.csv written before the crawler had it
        songs["instruments"] = grouped.instruments.agg(join_lists)
    songs["genre"] = songs.genres.str.split(";").str[0]
    return songs.reset_index(drop=True).drop(columns=["n"] + KEY_COLS)


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


def main():
    mb_raw = pd.concat([pd.read_csv(ROOT / f"songs_{y}.csv")
                        for y in range(FIRST_YEAR, LAST_YEAR + 1)], ignore_index=True)
    genius_raw = pd.read_csv(ROOT / "genius_songs_trimmed.csv")
    genius_raw = genius_raw[genius_raw.year.between(FIRST_YEAR, LAST_YEAR)]
    if genius_raw.year.nunique() < LAST_YEAR - FIRST_YEAR + 1:
        raise SystemExit(f"genius_songs_trimmed.csv lacks some of the years {YEARS}; "
                         f"re-run dataDownload/download-Genius.py")

    songs = clean_musicbrainz(mb_raw)
    print(f"MusicBrainz: {len(mb_raw)} recordings -> {len(songs)} songs")
    genius = clean_genius(genius_raw)
    misc = genius_raw.tag == "misc"
    foreign = ~misc & genius_raw.language.notna() & genius_raw.language.ne("en")
    print(f"Genius: {len(genius_raw)} pages -> {len(genius)} songs (dropped {misc.sum()} "
          f"non-song 'misc' pages, {foreign.sum()} non-English, "
          f"{len(genius_raw) - misc.sum() - foreign.sum() - len(genius)} duplicates/versions)")

    songs.to_csv(ROOT / f"songs_{YEARS}_clean.csv", index=False)
    genius.to_csv(ROOT / f"genius_{YEARS}_clean.csv", index=False)
    print(f"Wrote songs_{YEARS}_clean.csv, genius_{YEARS}_clean.csv")


if __name__ == "__main__":
    main()
