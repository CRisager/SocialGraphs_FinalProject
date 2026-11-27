"""
Attach Genius lyrics to the MusicBrainz songs, using the cleaned files from clean.py.

Matching (first rule that hits wins)
  1. normalised title + artist
  2. title without parentheses / version / part suffix + artist
  3. as 1 and 2, but with only the main artist ("Aretha Franklin with The Dixie Flyers")
  4. fuzzy title (similarity >= FUZZY_THRESHOLD) among the same artist's songs

Labels
  Each core genre (src/clean.py) belongs to one coarse class (COARSE_GENRES), and a
  song's MusicBrainz votes are summed per class:
    genre_coarse  the most-voted class among all of them (rock, pop, rb, country, folk,
                  blues, jazz, reggae, electronic, rap, other)
    mb_label      the most-voted class among the Genius tags only (pop, rock, rb,
                  country, rap), so it can be compared with the Genius tag; None when
                  no core genre of the song is in one of them
  Ties are drawn at random (seeded per song, see clean.pick_top). The merged `label` is
  the Genius tag when the song matched, else mb_label; `labels_agree` shows where both
  exist.

Output (data/processed/)
  songs_<years>_lyrics.csv   every MusicBrainz song, with lyrics where a match was found

Usage:
    python -m src.clean      # first, writes the *_clean.csv files
    python -m src.combine
"""

import re
from difflib import SequenceMatcher

import pandas as pd

from .clean import MATCH_RULES, add_keys, parse_votes, pick_top
from .config import GENIUS_CLEAN, SONGS_CLEAN, SONGS_LYRICS

FUZZY_THRESHOLD = 0.9  # below this, wrong pairs like "Waterloo" / "Water Boy" appear

# Coarse class of a core genre: the first class with a keyword that matches as whole
# word(s), so "rock" does not match "rocksteady" and the specific classes (rap, reggae)
# are tried before the broad ones. Core genres matching no class are "other".
COARSE_GENRES = {
    "rap": r"hip hop|rap|g-funk|trap|turntablism",
    "reggae": r"reggae|ska|rocksteady|dub|2 tone|dancehall|calypso|deejay|ragga",
    "rb": r"soul|r&b|funk|motown|disco|gospel|doo-wop|quiet storm|new jack swing|"
          r"boogie|hi-nrg|go-go|minneapolis sound",
    "country": r"country|countrypolitan|bluegrass|honky tonk|nashville sound|western|"
               r"bakersfield sound|americana|urban cowboy",
    "blues": r"blues|boogie-woogie",
    "jazz": r"jazz|swing|big band|\w*bop|post-bop|bossa nova|lounge|fusion|exotica|"
            r"ragtime|dixieland|third stream",
    "folk": r"folk|singer-songwriter|skiffle|celtic|old-time|cajun|sea shanty|jug band",
    "pop": r"pop|(?!britpop)\w+pop|ballad|easy listening|schlager|brill building|chanson|levenslied|"
           r"yé-yé|bubblegum|europop|nederpop|kayō|kayōkyoku|enka|sunshine|"
           r"canción melódica|tin pan alley",
    # synth and dance music; ahead of rock so "synthwave"/"coldwave" land here
    "electronic": r"electronic|electronica|electro|electroclash|dance|ambient|industrial|"
                  r"techno|house|trance|club|berlin school|minimal synth|synthwave|"
                  r"coldwave|ebm|idm|downtempo|chillout|breaks|breakbeat|jungle|dubstep|"
                  r"future bass|deep tech|leftfield|eurobeat|moogsploitation",
    "rock": r"rock|rockabilly|psychobilly|metal|nwobhm|punk|oi|psych\w*|neo-psychedelia|"
            r"\w*prog|avant-prog|progressive|canterbury scene|garage|beat music|beat|"
            r"freakbeat|merseybeat|mod|krautrock|glam|surf|jam band|wave|new romantic|"
            r"aor|grunge|nederbeat|palingsound|indorock|british invasion|no wave|"
            r"hardcore|britpop|shoegaze|gothic|deathrock|pigfuck|zolo|emo|queercore|"
            r"paisley underground",
}
_COARSE_RE = {c: re.compile(rf"(?<![\w-])(?:{kw})(?![\w-])") for c, kw in COARSE_GENRES.items()}
GENIUS_TAGS = ["pop", "rock", "rb", "country", "rap"]


def coarse_genre(core):
    return next((c for c, rx in _COARSE_RE.items() if rx.search(core)), "other")


for _core, _coarse in [("rock", "rock"), ("rocksteady", "reggae"), ("r&b", "rb"),
                       ("hip hop", "rap"), ("hard bop", "jazz"), ("doo-wop", "rb"),
                       ("singer-songwriter", "folk"), ("comedy", "other"),
                       ("psychedelia", "rock"), ("symphonic prog", "rock"),
                       ("post-bop", "jazz"), ("electropop", "pop"), ("synthwave", "electronic"),
                       ("rockabilly", "rock"), ("dance", "electronic"), ("britpop", "rock")]:
    assert coarse_genre(_core) == _coarse, (_core, coarse_genre(_core))


def class_votes(core_votes):
    """'rock:15;pop:1;mod:3' -> {'rock': 18, 'pop': 1}: core-genre votes summed per class."""
    votes = {}
    for core, n in parse_votes(core_votes).items():
        c = coarse_genre(core)
        votes[c] = votes.get(c, 0) + n
    return votes


assert class_votes("rock:15;pop:1;mod:3") == {"rock": 18, "pop": 1}


def coarse_labels(core_votes, song_id):
    """(genre_coarse, mb_label) of one song; see the module docstring."""
    votes = class_votes(core_votes)
    return (pick_top(votes, song_id),
            pick_top({c: v for c, v in votes.items() if c in GENIUS_TAGS}, song_id))


assert coarse_labels("jazz:3;soul:1", "x") == ("jazz", "rb")
assert coarse_labels("comedy:2", "x") == ("other", None)
assert coarse_labels(None, "x") == (None, None)


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


def run(songs=None, genius=None):
    """Match the cleaned datasets, write songs_<years>_lyrics.csv and return it."""
    songs = pd.read_csv(SONGS_CLEAN) if songs is None else songs
    genius = pd.read_csv(GENIUS_CLEAN) if genius is None else genius

    found = match(add_keys(songs, "song", "artist"), add_keys(genius, "title", "artist"))
    lyrics = (genius[["title", "artist", "tag", "language", "lyrics"]]
              .rename(columns={"title": "genius_title", "artist": "genius_artist",
                               "tag": "genius_tag", "language": "lyrics_language"}))
    combined = (songs.join(found).join(lyrics, on="genius_index")
                .drop(columns="genius_index"))
    combined["genre_coarse"], combined["mb_label"] = zip(
        *[coarse_labels(v, i) for v, i in zip(combined.core_votes, combined.id)])
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

    combined.to_csv(SONGS_LYRICS, index=False)
    print(f"\nWrote {SONGS_LYRICS.name}")
    return combined


if __name__ == "__main__":
    run()
