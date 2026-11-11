"""
Attach Genius lyrics to the MusicBrainz songs, using the cleaned files from clean.py.

Matching (first rule that hits wins)
  1. normalised title + artist
  2. title without parentheses / version / part suffix + artist
  3. as 1 and 2, but with only the main artist ("Aretha Franklin with The Dixie Flyers")
  4. fuzzy title (similarity >= FUZZY_THRESHOLD) among the same artist's songs

Labels
  MusicBrainz genres are mapped onto the Genius tags (pop, rock, country, rb, rap) by
  keyword votes over the song's genre list (mb_label). The merged `label` is the Genius
  tag when the song matched, else mb_label; `labels_agree` shows where both exist.

Output (repo root)
  songs_<years>_lyrics.csv   every MusicBrainz song, with lyrics where a match was found

Usage:
    python dataCleaner/clean.py      # first, writes the *_clean.csv files
    python dataCleaner/combine.py
"""

import re
from difflib import SequenceMatcher

import pandas as pd

from clean import MATCH_RULES, ROOT, YEARS, add_keys

FUZZY_THRESHOLD = 0.9  # below this, wrong pairs like "Waterloo" / "Water Boy" appear

# MusicBrainz genre keywords per Genius tag; "country rock" votes for both country and rock.
# Genres with no keyword (jazz, blues, folk, reggae, classical, ...) cast no vote.
LABEL_KEYWORDS = {
    "country": r"country|bluegrass|honky tonk|nashville|western|bakersfield|americana|urban cowboy",
    "rb": r"soul|r&b|rhythm & blues|funk|motown|disco|gospel|doo-wop|quiet storm|new jack swing",
    "pop": r"pop|ballad|easy listening|schlager|brill building",
    "rock": r"rock|metal|punk|psych|prog|garage|beat|mod$|krautrock|glam|surf|jam band",
    "rap": r"hip hop|rap",
}


def genre_label(genres, primary):
    """Genius tag with most keyword hits in a ';'-joined MusicBrainz genre list; the
    primary genre breaks ties. None if no genre maps."""
    if not isinstance(genres, str):
        return None
    votes = {tag: sum(bool(re.search(kw, g)) for g in genres.split(";"))
             for tag, kw in LABEL_KEYWORDS.items()}
    best = max(votes.values())
    if best == 0:
        return None
    tied = [tag for tag, v in votes.items() if v == best]
    return next((tag for tag in tied if re.search(LABEL_KEYWORDS[tag], str(primary))), tied[0])


assert genre_label("country rock;country;rock", "country rock") == "country"
assert genre_label("soul;motown;doo-wop;r&b", "soul") == "rb"
assert genre_label("pop rock", "pop rock") == "pop"  # tie: first key in LABEL_KEYWORDS wins
assert genre_label("jazz;blues", "jazz") is None


def match(songs, genius):
    """Index of each song's Genius row (None if not found) and the rule that found it."""
    lookups = [dict(zip(zip(genius[t], genius[a]), genius.index)) for (t, a), _ in MATCH_RULES]
    by_artist = dict(list(genius.groupby("artist_key")))
    by_main = dict(list(genius.groupby("artist_main")))

    def find(song):
        for ((t, a), rule), lookup in zip(MATCH_RULES, lookups):
            if (song[t], song[a]) in lookup:
                return lookup[(song[t], song[a])], rule
        candidates = by_artist.get(song.artist_key, by_main.get(song.artist_main))
        if candidates is not None:
            ratio, j = max((SequenceMatcher(None, song.title_core, c).ratio(), j)
                           for j, c in candidates.title_core.items())
            if ratio >= FUZZY_THRESHOLD:
                return j, "fuzzy title"
        return None, None

    return pd.DataFrame([find(s) for _, s in songs.iterrows()],
                        index=songs.index, columns=["genius_index", "match"])


def main():
    songs = pd.read_csv(ROOT / f"songs_{YEARS}_clean.csv")
    genius = pd.read_csv(ROOT / f"genius_{YEARS}_clean.csv")

    found = match(add_keys(songs, "song", "artist"), add_keys(genius, "title", "artist"))
    lyrics = (genius[["title", "artist", "tag", "language", "lyrics"]]
              .rename(columns={"title": "genius_title", "artist": "genius_artist",
                               "tag": "genius_tag", "language": "lyrics_language"}))
    combined = (songs.join(found).join(lyrics, on="genius_index")
                .drop(columns="genius_index"))
    combined["mb_label"] = [genre_label(g, p) for g, p in zip(combined.genres, combined.genre)]
    combined["label"] = combined.genius_tag.fillna(combined.mb_label)
    both = combined.genius_tag.notna() & combined.mb_label.notna()
    combined["labels_agree"] = (combined.genius_tag == combined.mb_label).where(both)

    print(f"Matched {combined.match.notna().sum()} of {len(songs)} songs "
          f"({combined.match.notna().mean():.0%}):")
    print(combined.match.value_counts().to_string())
    shared = found.genius_index.dropna().duplicated(keep=False).sum()
    if shared:
        print(f"Note: {shared} songs share their Genius page with another song")

    print(f"\nLabels: {combined.label.notna().sum()} of {len(songs)} songs labelled")
    print(combined.label.value_counts().to_string())
    print(f"Genius and MusicBrainz agree on {combined.labels_agree.mean():.0%} "
          f"of the {both.sum()} songs labelled by both")

    combined.to_csv(ROOT / f"songs_{YEARS}_lyrics.csv", index=False)
    print(f"\nWrote songs_{YEARS}_lyrics.csv")


if __name__ == "__main__":
    main()
