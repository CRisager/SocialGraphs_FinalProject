"""
Look up the MusicBrainz songs of one year (songs_<year>.csv) on Discogs and collect
their credits (instruments, producers, singers, labels), to compare credit coverage
with MusicBrainz (see dataExploration/credit_coverage.py).

Matching (per song; the first search whose releases contain the track wins)
  1. Discogs release search with artist + title
  2. title without parentheses ("(Live)", "(Part 1)")
  3. only the main artist ("Aretha Franklin with The Dixie Flyers" -> "Aretha Franklin")
  4. main artist + title without parentheses
  The releases a search returns are opened closest-to-the-year first (one per master,
  at most MAX_FETCHES new lookups per song). A release counts only if its tracklist has
  the song: normalised titles equal, or similarity >= TITLE_THRESHOLD.

Credits
  On Discogs a credit sits on the track itself, or on the release with a "tracks" field
  naming the tracks it covers ("A1 to A3, B2"): both are kept as level "track". Release
  credits without a "tracks" field are for the whole release: level "album". Each role
  ("Guitar [Lead], Vocals") is classed by keyword as producer, vocals, musician (an
  instrument) or other (written-by, engineer, artwork, ...).

Output (repo root)
  discogs_<year>.csv              one row per MusicBrainz song; `id` joins songs_<year>.csv
  discogs_credits_<year>.csv      one row per credit and role (other roles included)
  discogs_<year>_checkpoint.json  resume state: stop with Ctrl+C, re-run to continue
  collection_log.jsonl            one line per run (source "discogs"), for Methods

Discogs allows 60 requests/minute with a personal token: expect 1.5-2.5 hours for
~2,000 songs.

Usage:
    export DISCOGS_TOKEN=...   # discogs.com > Settings > Developers > Generate new token
    python dataDownload/crawler-discogs.py              # 1970
    python dataDownload/crawler-discogs.py --year 1971
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import requests

API = "https://api.discogs.com"
ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "SocialGraphsCourseProject/1.0 +https://sunelehmann.com/socialgraphs2026-web"}
RATE_LIMIT_SECONDS = 1.05  # authenticated: 60 requests/minute
TITLE_THRESHOLD = 0.9      # same as the Genius matching in combine.py
MAX_FETCHES = 3            # new release lookups per song
SAVE_EVERY = 25            # songs between checkpoint writes

# Role keywords, tested in this order on the role without its [modifiers].
PRODUCER = r"produc"
VOCALS = r"vocal|voice|choir|chorus"
OTHER = (r"arrang|conduct|written|compos|lyric|music by|words by|engineer|mix|master|"
         r"record|design|photo|artwork|art direction|liner|notes|management|supervis|"
         r"edit|lacquer|cut by|transfer|copyist|orchestrat|directed")
INSTRUMENT = (r"guitar|bass|drum|percussion|piano|organ|keyboard|synth|moog|mellotron|"
              r"harpsichord|clavinet|celest|violin|viola|cello|fiddle|string|horn|trumpet|"
              r"trombone|tuba|cornet|flugelhorn|sax|flute|piccolo|clarinet|oboe|bassoon|"
              r"harmonica|banjo|mandolin|steel|dobro|harp|tambourine|conga|bongo|vibraphone|"
              r"xylophone|marimba|glockenspiel|sitar|tabla|accordion|ukulele|bells|timpani|"
              r"cymbal|instrument|whistle|kazoo|washboard|zither|dulcimer|lute|koto|"
              r"shaker|cowbell|triangle|brass|woodwind|reeds|electronics|orchestra|band|"
              r"musician|ensemble|accompan")
ROLES = {"producers": "producer", "musicians": "musician", "vocals": "vocals"}

session = requests.Session()
session.headers.update(HEADERS)
_last_call = 0.0
stats = {"requests": 0, "retries": 0}


def set_year(year):
    global YEAR, CHECKPOINT
    YEAR = int(year)
    CHECKPOINT = ROOT / f"discogs_{YEAR}_checkpoint.json"


def get(path, params=None):
    """GET with rate limiting; retries 429/5xx and network errors. None on 404."""
    global _last_call
    for attempt in range(10):
        wait = RATE_LIMIT_SECONDS - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        backoff = 60 if attempt == 0 else min(2 ** attempt, 120)
        stats["requests"] += 1
        stats["retries"] += attempt > 0
        try:
            r = session.get(API + path, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  network error ({e}), retrying in {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            print(f"  HTTP {r.status_code}, retrying in {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
            continue
        r.raise_for_status()  # 401: token missing or wrong
    raise RuntimeError(f"Gave up on {path} {params}")


def load_state():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"songs": {}, "releases": {}}


def save_state(state):
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, CHECKPOINT)


# ---------- matching ----------

def norm(title):
    """Lowercase, no parentheses (unless that leaves nothing), letters and digits only."""
    t = title.lower().replace("’", "'").replace("&", "and")
    core = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    return re.sub(r"[^a-z0-9]+", " ", core if core.strip() else t).strip()


def same_title(a, b):
    a, b = norm(a), norm(b)
    return a == b or SequenceMatcher(None, a, b).ratio() >= TITLE_THRESHOLD


def main_artist(artist):
    return re.split(r"\s+(?:&|and|with|feat\.?|featuring|ft\.?|vs\.?|/)\s+|,\s+",
                    artist, maxsplit=1)[0]


def queries(artist, title):
    core = re.sub(r"\s*[(\[].*?[)\]]", "", title).strip() or title
    main = main_artist(artist)
    rules = [("artist + title", artist, title), ("core title", artist, core),
             ("main artist", main, title), ("main artist + core title", main, core)]
    seen, out = set(), []
    for rule, a, t in rules:
        if (a, t) not in seen:
            seen.add((a, t))
            out.append((rule, a, t))
    return out


def search(artist, title):
    """Release ids for artist + track, closest to YEAR first, one per master."""
    data = get("/database/search", {"artist": artist, "track": title,
                                    "type": "release", "per_page": 50}) or {}
    year = lambda r: int(r["year"]) if str(r.get("year", "")).isdigit() else 9999
    ids, masters = [], set()
    for r in sorted(data.get("results", []), key=lambda r: (abs(year(r) - YEAR), year(r))):
        if r.get("master_id") and r["master_id"] in masters:
            continue
        masters.add(r.get("master_id"))
        ids.append(str(r["id"]))
    return ids


def trim(release):
    """The parts of a /releases/{id} response we keep."""
    credit = lambda c: [c.get("id"), re.sub(r" \(\d+\)$", "", c.get("name", "")),
                        c.get("role", ""), c.get("tracks", "")]
    tracks = []
    for t in release.get("tracklist", []):
        for s in (t.get("sub_tracks") or [t]) if t.get("type_") == "index" else [t]:
            if s.get("type_", "track") == "track":
                tracks.append([s.get("position", ""), s.get("title", ""),
                               [credit(c) for c in s.get("extraartists", [])]])
    return {"title": release.get("title", ""), "year": release.get("year"),
            "extraartists": [credit(c) for c in release.get("extraartists", [])],
            "tracklist": tracks,
            "labels": [[l.get("id"), l.get("name", "")] for l in release.get("labels", [])],
            "genres": release.get("genres", []), "styles": release.get("styles", [])}


def find_track(release, title):
    return next((i for i, (_, t, _) in enumerate(release["tracklist"]) if same_title(t, title)),
                None)


def match_song(state, artist, title):
    fetches = 0
    for rule, a, t in queries(artist, title):
        for rid in search(a, t):
            if rid not in state["releases"]:
                if fetches == MAX_FETCHES:
                    return None
                fetches += 1
                rel = get(f"/releases/{rid}")
                state["releases"][rid] = trim(rel) if rel else None
            rel = state["releases"][rid]
            i = find_track(rel, title) if rel else None
            if i is not None:
                return {"rule": rule, "release": rid, "track": i}
    return None


# ---------- credits ----------

def covers(tracks_field, pos, positions):
    """Does a release credit's "tracks" field ("A1 to A3, B2") include position pos?"""
    for part in (p.strip() for p in tracks_field.split(",")):
        if " to " in part:
            a, b = (x.strip() for x in part.split(" to ", 1))
            if a in positions and b in positions and \
                    positions.index(a) <= positions.index(pos) <= positions.index(b):
                return True
        elif part and part == pos:
            return True
    return False


def split_roles(role):
    """'Guitar [Lead, Slide], Vocals' -> ['Guitar [Lead, Slide]', 'Vocals']"""
    return [r.strip() for r in re.split(r",(?![^\[]*\])", role) if r.strip()]


def classify(role):
    base = re.sub(r"\s*\[.*?\]", "", role).strip().lower()
    for kind, pattern in (("producer", PRODUCER), ("vocals", VOCALS), ("other", OTHER),
                          ("musician", INSTRUMENT)):
        if re.search(pattern, base):
            return kind, base
    return "other", base


def song_credits(release, i):
    """[level, kind, role, discogs artist id, name] for track i of the release."""
    positions = [p for p, _, _ in release["tracklist"]]
    pos, _, own = release["tracklist"][i]
    found = [("track", c) for c in own]
    for c in release["extraartists"]:
        if not c[3]:
            found.append(("album", c))
        elif covers(c[3], pos, positions):
            found.append(("track", c))
    return [[level, *classify(r), aid, name]
            for level, (aid, name, role, _) in found for r in split_roles(role)]


def names(credits, kind, level, field=4):
    return sorted({c[field] for c in credits if c[1] == kind and c[0] == level})


# a fake release: a producer for the whole release, a drummer on A1-A2 only, a
# guitarist and a backing singer on the B-side's own track entry
_rel = trim({"extraartists": [
    {"id": 1, "name": "Prod (2)", "role": "Producer", "tracks": ""},
    {"id": 2, "name": "Drm", "role": "Drums", "tracks": "A1 to A2"}],
    "tracklist": [{"position": "A1", "title": "Rats"}, {"position": "A2", "title": "Rat"},
                  {"position": "B", "title": "Sing a Song for Freedom (Single Version)",
                   "extraartists": [{"id": 3, "name": "G", "role": "Guitar [Lead, Slide], Backing Vocals"},
                                    {"id": 4, "name": "W", "role": "Written-By, Arranged By [Strings]"}]}]})
assert find_track(_rel, "Rats") == 0 and find_track(_rel, "Sing a Song for Freedom") == 2
_a2, _b = song_credits(_rel, 1), song_credits(_rel, 2)
assert names(_a2, "musician", "track", 2) == ["drums"] and names(_a2, "producer", "album") == ["Prod"]
assert names(_b, "musician", "track", 2) == ["guitar"] and names(_b, "vocals", "track") == ["G"]
assert names(_b, "musician", "track") == ["G"]  # written-by / arranged by [strings] are "other"
assert classify("Bass Vocals")[0] == "vocals" and classify("Recorded By")[0] == "other"
assert main_artist("Aretha Franklin with The Dixie Flyers") == "Aretha Franklin"
del _rel, _a2, _b


# ---------- run ----------

def song_rows(state, songs):
    """One output row per MusicBrainz song, plus all its credits."""
    for s in songs:
        m = state["songs"].get(s["id"])
        rel = state["releases"][m["release"]] if m else None
        credits = song_credits(rel, m["track"]) if m else []
        cols = {}
        for prefix, level in (("", "track"), ("album_", "album")):
            cols[prefix + "instruments"] = names(credits, "musician", level, 2)  # roles
            for col, kind in ROLES.items():
                cols[prefix + col] = names(credits, kind, level)  # people
        yield s, m, rel, credits, cols


def crawl(songs):
    state = load_state()
    todo = [s for s in songs if s["id"] not in state["songs"]]
    print(f"{len(todo)} of {len(songs)} songs left (~{len(todo) * 2.7 / 60:.0f} min)")
    for n, s in enumerate(todo, 1):
        state["songs"][s["id"]] = match_song(state, s["artist"], s["song"])
        if n % SAVE_EVERY == 0 or n == len(todo):
            save_state(state)
            done = state["songs"].values()
            print(f"  {n}/{len(todo)}  matched so far: {sum(map(bool, done))}/{len(done)}")


def write_outputs(songs, state=None):
    state = state or load_state()
    cols = ["id", "song", "artist", "match", "discogs_release_id", "discogs_release",
            "discogs_year", "track_position", "instruments", "album_instruments",
            "producers", "album_producers", "musicians", "album_musicians", "vocals",
            "album_vocals", "labels", "discogs_genres", "discogs_styles"]
    with open(ROOT / f"discogs_{YEAR}.csv", "w", encoding="utf-8", newline="") as f, \
            open(ROOT / f"discogs_credits_{YEAR}.csv", "w", encoding="utf-8", newline="") as g:
        w, wc = csv.writer(f), csv.writer(g)
        w.writerow(cols)
        wc.writerow(["recording_id", "discogs_release_id", "level", "kind", "role",
                     "discogs_artist_id", "name"])
        for s, m, rel, credits, c in song_rows(state, songs):
            j = lambda xs: ";".join(map(str, xs))
            w.writerow([s["id"], s["song"], s["artist"], m["rule"] if m else "",
                        m["release"] if m else "", rel["title"] if m else "",
                        rel["year"] if m else "", rel["tracklist"][m["track"]][0] if m else "",
                        *(j(c[k]) for k in cols[8:16]),
                        j(sorted({n for _, n in rel["labels"]})) if m else "",
                        j(rel["genres"]) if m else "", j(rel["styles"]) if m else ""])
            for credit in credits:
                wc.writerow([s["id"], m["release"], *credit])
    print(f"Wrote discogs_{YEAR}.csv and discogs_credits_{YEAR}.csv")


def coverage(songs, state):
    """Match and credit counts for the collection log."""
    rows = list(song_rows(state, songs))
    has = lambda test: sum(bool(test(c)) for _, m, _, _, c in rows if m)
    rules = {}
    for _, m, *_ in rows:
        if m:
            rules[m["rule"]] = rules.get(m["rule"], 0) + 1
    return {
        "songs_input": len(songs), "songs_looked_up": len(state["songs"]),
        "songs_matched": sum(bool(m) for _, m, *_ in rows), "matched_by_rule": rules,
        "songs_with_track_instruments": has(lambda c: c["instruments"]),
        "songs_with_any_instruments": has(lambda c: c["instruments"] or c["album_instruments"]),
        "songs_with_track_producers": has(lambda c: c["producers"]),
        "songs_with_any_producers": has(lambda c: c["producers"] or c["album_producers"]),
        "songs_with_any_vocals": has(lambda c: c["vocals"] or c["album_vocals"]),
    }


def write_log(songs, started, completed):
    state = load_state()
    entry = {
        "source": "discogs", "year": YEAR, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed": completed, "endpoint": API,
        "search": "GET /database/search type=release artist=<artist> track=<title>, "
                  "then GET /releases/{id}",
        "title_threshold": TITLE_THRESHOLD, "max_release_fetches_per_song": MAX_FETCHES,
        "rate_limit_s_per_request": RATE_LIMIT_SECONDS,
        "requests_this_run": stats["requests"], "retried_requests_this_run": stats["retries"],
        "releases_fetched": len(state["releases"]), **coverage(songs, state),
    }
    with open(ROOT / "collection_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=1970)
    args = ap.parse_args()
    set_year(args.year)
    token = os.environ.get("DISCOGS_TOKEN")
    if not token:
        sys.exit("Set DISCOGS_TOKEN first (discogs.com > Settings > Developers).")
    session.headers["Authorization"] = f"Discogs token={token}"
    with open(ROOT / f"songs_{YEAR}.csv", encoding="utf-8") as f:
        songs = list(csv.DictReader(f))
    started, completed = datetime.now(timezone.utc).isoformat(timespec="seconds"), False
    try:
        crawl(songs)
        completed = True
    finally:  # also log and write an interrupted run
        write_outputs(songs)
        write_log(songs, started, completed)


if __name__ == "__main__":
    main()
