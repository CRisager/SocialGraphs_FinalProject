"""
Count English-language songs first released in a given year (or range of years),
using the MusicBrainz API. Each year is crawled separately, with its own checkpoint
(data/raw/mb_<year>_checkpoint.json) and output (data/raw/songs_<year>.csv).

Definition used (adjust to taste):
  * "release"  = an official release dated YEAR whose tracklist language is English (lang:eng)
  * "song"     = a distinct MusicBrainz *recording* whose first-release-date is in YEAR
                 (so reissues of earlier tracks on YEAR compilations are not counted)
  * all genres  - no genre filter is applied, but each song is labelled with genres:
                 the recording's own genre tags if it has any, otherwise its artists'.
                 Only tags on MusicBrainz's official genre list count (so not "70s").
  * credits     - each song is also labelled with who worked on it: producers,
                 musicians (and the instruments they played), singers (lead and
                 backing; the attributes in credits_<year>.csv tell them apart) and
                 record labels. Credits on the recording itself are used where it has
                 any, otherwise those on its releases. Most recordings have none:
                 MusicBrainz only knows what editors entered.

Four phases:
  1. Search  - pages through /ws/2/release search (100 per request) to list candidate releases.
               This alone gives a quick upper bound (sum of track counts).
  2. Lookup  - fetches the recordings of ~45 releases per request via the recording search
               and dedupes them. MusicBrainz allows ~1 request/second; progress is
               checkpointed, so you can stop (Ctrl+C) and re-run to resume.
  3. Genres  - fetches the tags of ~45 artists per request via the artist search.
  4. Credits - one lookup per release (relationships are not in the search index),
               so this is the slow phase: ~1.1 s per release, about half an hour for
               1,600 releases. Skip it with --no-credits.

Output per year (data/raw/)
  songs_<year>.csv    one row per song; credit columns hold ';'-joined names
  credits_<year>.csv  one row per credit (song, level, type, attributes, MusicBrainz id,
                      name), labels included: what the collaboration graph is built from
  results/collection_log.jsonl  one line per run: endpoint, query, dates, request count, rate
                      limit and resulting counts (for the Methods section)

Usage:
    python -m src.download.musicbrainz                    # all phases, years from config.py
    python -m src.download.musicbrainz --years 1970 1970  # a single year
    python -m src.download.musicbrainz --quick            # phase 1 only (fast upper bound)
    python -m src.download.musicbrainz --no-credits       # phases 1-3 only (minutes, not hours)
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

from ..config import (COLLECTION_LOG, FIRST_YEAR, LAST_YEAR, RAW_DIR, musicbrainz_checkpoint,
                      musicbrainz_credits, musicbrainz_raw)

API = "https://musicbrainz.org/ws/2"
# MusicBrainz requires a descriptive User-Agent with contact info, or it will block you.
HEADERS = {"User-Agent": "SongsCounter/1.0 ( 2maasdumai@gmail.com )"}
RATE_LIMIT_SECONDS = 1.1
LANG = "eng"  # ISO 639-3 code of the tracklist language


def set_year(year):
    """The functions below crawl one year at a time, set here."""
    global YEAR, CHECKPOINT, QUERY
    YEAR = str(year)
    CHECKPOINT = str(musicbrainz_checkpoint(YEAR))
    # Lucene query for the release search. Drop the NOT clause to include compilations.
    QUERY = (f'date:{YEAR} AND lang:{LANG} AND status:official AND type:single '
             f'AND NOT secondarytype:compilation')


set_year(1970)

session = requests.Session()
session.headers.update(HEADERS)
_last_call = 0.0
stats = {"requests": 0, "retries": 0}  # for the collection log


def get(path, params):
    """GET with rate limiting and retry on 503/429 (MusicBrainz's 'slow down' signal)
    and other temporary server errors. Backs off up to a minute per try (~8 min total)."""
    global _last_call
    params = {"fmt": "json", **params}
    for attempt in range(12):
        wait = RATE_LIMIT_SECONDS - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        backoff = min(2 ** attempt, 60)
        stats["requests"] += 1
        stats["retries"] += attempt > 0
        try:
            r = session.get(f"{API}/{path}", params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  network error ({e}), retrying in {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
            continue
        if r.status_code == 200:
            return r.json() if params["fmt"] == "json" else r.text
        if r.status_code in (429, 500, 502, 503, 504):
            print(f"  HTTP {r.status_code}, retrying in {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Gave up on {path} {params}")


def load_state():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"releases": {}, "search_offset": 0, "search_total": None,
            "done_releases": [], "recordings": {}, "artists": {},
            "recording_credits": {}, "release_credits": {}}


def save_state(state):
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, CHECKPOINT)


# The search API refuses offset >= 500 (HTTP 400). So when a query matches more
# than that, it is split into smaller sub-queries that together cover exactly the
# same releases (each split ends with a "none of the above" bucket).
# Values are queried in OR-groups that are only halved while they exceed the cap,
# so sparse/empty values cost one shared request instead of one request each.
# Country goes first: it splits far better than the others (e.g. 1975 singles are
# almost all 2 tracks, and date ranges don't help since year-only dates match all).
PAGE_CAP = 500
LETTERS = "abcdefghijklmnopqrstuvwxyz0123456789"
SPLITTERS = [
    ("country", ["US", "GB", "CA", "AU", "NZ", "IE", "XE", "XW", "DE", "NL",
                 "FR", "JP", "SE", "IT", "ES", "BR", "ZA", "JM", "NO", "DK"]),
    ("primarytype", ["album", "single", "ep", "broadcast", "other"]),
    ("tracks", [str(n) for n in range(1, 41)]),
    # US and GB 2-track singles alone pass the cap from 1978 on, and most are dated by
    # year only. So split on the first letter of any word in the artist credit, then in
    # the title. A release can match several letters; they are keyed by id, so it is
    # only stored once.
    ("artist", [c + "*" for c in LETTERS]),
    ("release", [c + "*" for c in LETTERS]),
]


def or_clause(field, values):
    return " OR ".join(f"{field}:{v}" for v in values)


def add_releases(state, data):
    for rel in data.get("releases", []):
        state["releases"][rel["id"]] = {
            "title": rel.get("title"),
            "date": rel.get("date"),
            "track_count": rel.get("track-count", 0),
            "country": rel.get("country"),
            "language": (rel.get("text-representation") or {}).get("language"),
        }


def crawl(q, state, depth=0, first=None):
    if q in state["done_queries"]:
        return
    if first is None:
        first = get("release", {"query": q, "limit": 100, "offset": 0})
    count = first["count"]
    if depth == 0:
        state["search_total"] = count

    if count > PAGE_CAP and depth < len(SPLITTERS):
        field, values = SPLITTERS[depth]
        print(f"  {'  ' * depth}{count} results > {PAGE_CAP}, splitting by {field}")
        crawl_group(q, field, values, state, depth)
        # everything not covered by the listed values
        crawl(f"({q}) AND NOT ({or_clause(field, values)})", state, depth + 1)
    else:
        if count > PAGE_CAP:
            print(f"  WARNING: slice still has {count} results; only the first "
                  f"{PAGE_CAP} can be fetched: {q}", file=sys.stderr)
        add_releases(state, first)
        offset = 100
        while offset < min(count, PAGE_CAP):
            data = get("release", {"query": q, "limit": 100, "offset": offset})
            add_releases(state, data)
            offset += 100
        print(f"  {'  ' * depth}slice done ({count}); total collected: "
              f"{len(state['releases'])}")

    state["done_queries"].append(q)
    save_state(state)


def crawl_group(q, field, values, state, depth):
    """Fetch q restricted to any of `values`, halving the group while it's too big."""
    sub = f"({q}) AND ({or_clause(field, values)})"
    if sub in state["done_queries"]:
        return
    first = get("release", {"query": sub, "limit": 100, "offset": 0})
    if first["count"] <= PAGE_CAP or len(values) == 1:
        # small enough to page through, or a single value -> split by next field
        crawl(sub, state, depth + 1, first)
        return
    mid = len(values) // 2
    crawl_group(q, field, values[:mid], state, depth)
    crawl_group(q, field, values[mid:], state, depth)
    state["done_queries"].append(sub)
    save_state(state)


def phase1_search(state):
    """Collect all releases matching QUERY, splitting around the paging cap."""
    if state.get("splitters") != len(SPLITTERS):
        # new year, or one searched with fewer splitters (slices may have been cut at
        # the paging cap): search again, releases already found are kept
        state["done_queries"], state["splitters"] = [], len(SPLITTERS)
    crawl(QUERY, state)


# Instead of one /release lookup per release, ask the recording search for the
# recordings on many releases at once (reid:(a OR b OR ...)). Each result already
# carries first-release-date and artist-credit, so ~45 singles cost one request.
# Releases with many tracks just mean a few extra pages (offset cap is 500).
LOOKUP_BATCH = 45


def release_origin(rec, releases):
    """Country and language of the earliest of our releases this recording is on."""
    ours = [r for r in rec.get("releases", []) if r["id"] in releases]
    if not ours:
        return None, LANG
    rel = min(ours, key=lambda r: (not r.get("country"), r.get("date") or "9999"))
    # older checkpoints didn't store language; every release matched lang:LANG anyway
    language = releases[rel["id"]].get("language") or LANG
    return rel.get("country"), language


def phase2_lookup(state):
    """Fetch recordings of the found releases in batches; keep those first released in YEAR."""
    done = set(state["done_releases"])
    todo = [rid for rid in state["releases"] if rid not in done]
    n_req = -(-len(todo) // LOOKUP_BATCH)
    print(f"  lookup: {len(todo)} releases left (~{n_req} requests, "
          f"~{n_req * RATE_LIMIT_SECONDS / 60:.0f} min)")
    for b in range(0, len(todo), LOOKUP_BATCH):
        batch = todo[b:b + LOOKUP_BATCH]
        q = f"reid:({' OR '.join(batch)})"
        offset, count = 0, 1
        while offset < min(count, PAGE_CAP):
            data = get("recording", {"query": q, "limit": 100, "offset": offset})
            count = data["count"]
            for rec in data.get("recordings", []):
                first = rec.get("first-release-date") or ""
                if first.startswith(YEAR) and rec.get("id"):
                    credits = rec.get("artist-credit", [])
                    artist = "".join(
                        ac.get("name", "") + ac.get("joinphrase", "") for ac in credits
                    )
                    artist_ids = [ac["artist"]["id"] for ac in credits if "artist" in ac]
                    tags = [[t["name"], t.get("count", 0)] for t in rec.get("tags", [])]
                    country, language = release_origin(rec, state["releases"])
                    state["recordings"][rec["id"]] = [rec.get("title"), artist,
                                                      artist_ids, tags, first[:4],
                                                      country, language]
            offset += 100
        if count > PAGE_CAP:
            print(f"  WARNING: batch had {count} recordings, only {PAGE_CAP} fetched",
                  file=sys.stderr)
        state["done_releases"].extend(batch)
        save_state(state)
        print(f"  lookup: {min(b + LOOKUP_BATCH, len(todo))}/{len(todo)}  "
              f"distinct {YEAR} recordings so far: {len(state['recordings'])}")


# Artist search results include folksonomy tags with vote counts. Genres in
# MusicBrainz are just tags that match the official genre list, so filtering the
# tags by that list gives the same genres as a per-artist ?inc=genres lookup,
# at ~45 artists per request instead of one.
ARTIST_BATCH = 45


def fetch_genre_names():
    return set(get("genre/all", {"fmt": "txt"}).strip().splitlines())


def phase3_genres(state):
    """Fetch tags for every artist credited on a kept recording."""
    artists = state.setdefault("artists", {})
    todo = sorted({aid for rec in state["recordings"].values() for aid in rec[2]}
                  - artists.keys())
    n_req = -(-len(todo) // ARTIST_BATCH)
    print(f"  genres: {len(todo)} artists left (~{n_req} requests, "
          f"~{n_req * RATE_LIMIT_SECONDS / 60:.0f} min)")
    for b in range(0, len(todo), ARTIST_BATCH):
        batch = todo[b:b + ARTIST_BATCH]
        data = get("artist", {"query": f"arid:({' OR '.join(batch)})", "limit": 100})
        for aid in batch:
            artists[aid] = []  # mark as done even if it has no tags
        for a in data.get("artists", []):
            artists[a["id"]] = [[t["name"], t.get("count", 0)] for t in a.get("tags", [])]
        save_state(state)
        print(f"  genres: {min(b + ARTIST_BATCH, len(todo))}/{len(todo)}")


def song_genres(rec, artists, genre_names):
    """Genres of a recording, most-voted first: its own if tagged, else its artists'."""
    def pick(tags):
        votes = {}
        for name, count in tags:
            if name in genre_names and count > 0:
                votes[name] = votes.get(name, 0) + count
        return sorted(votes, key=votes.get, reverse=True)

    artist_ids, tags = rec[2], rec[3]
    return pick(tags) or pick([t for aid in artist_ids for t in artists.get(aid, [])])


# Who worked on what is stored as relationships, which no search index contains, so
# unlike phases 2 and 3 this cannot be batched: one lookup per release. It returns the
# artist relationships of every recording on the release (X played guitar, Y produced)
# and of the release as a whole, plus the release's labels.
CREDITS_INC = "recordings+artist-rels+recording-level-rels+labels"
NOT_INSTRUMENTS = {"additional", "guest", "solo"}  # the relationship's other attributes
SAVE_EVERY = 25  # lookups between checkpoint writes
# credit columns of songs_<year>.csv: which relationship types each one collects
ROLES = {"producers": {"producer"}, "musicians": {"instrument"}, "vocals": {"vocal"}}


def artist_rels(entity):
    """[type, artist id, artist name, attributes] of every artist credited on entity."""
    return [[rel["type"], rel["artist"]["id"], rel["artist"]["name"], rel.get("attributes", [])]
            for rel in entity.get("relations", []) if "artist" in rel]


def release_labels(release):
    return [[li["label"]["id"], li["label"]["name"]]
            for li in release.get("label-info", []) if li.get("label")]


def instruments_of(rels):
    return sorted({a for t, _, _, attrs in rels if t == "instrument"
                   for a in attrs if a not in NOT_INSTRUMENTS})


def phase4_credits(state):
    """Fetch credits. recording_credits: recording id -> its own artist relationships.
    release_credits: release id -> [its artist relationships, its labels,
    the ids of our recordings on it]."""
    rec_credits = state.setdefault("recording_credits", {})
    rel_credits = state.setdefault("release_credits", {})
    todo = [rid for rid in state["releases"] if rid not in rel_credits]
    print(f"  credits: {len(todo)} releases left "
          f"(~{len(todo) * RATE_LIMIT_SECONDS / 60:.0f} min)")
    for i, rid in enumerate(todo, 1):
        try:
            rel = get(f"release/{rid}", {"inc": CREDITS_INC})
        except requests.HTTPError as e:
            if e.response.status_code != 404:
                raise
            rel = {}  # release was deleted after the search found it
        ours = []
        for medium in rel.get("media", []):
            for track in medium.get("tracks", []):
                rec = track.get("recording", {})
                if rec.get("id") in state["recordings"]:
                    rec_credits[rec["id"]] = artist_rels(rec)
                    ours.append(rec["id"])
        rel_credits[rid] = [artist_rels(rel), release_labels(rel), ours]
        if i % SAVE_EVERY == 0 or i == len(todo):
            save_state(state)
            print(f"  credits: {i}/{len(todo)}  recordings with credits so far: "
                  f"{sum(map(bool, rec_credits.values()))}")


def song_credits(state):
    """Per recording: (its own relationships, those of its releases, its labels)."""
    shared = {}
    for rels, labels, ours in state.get("release_credits", {}).values():
        for rec_id in ours:
            r, l = shared.setdefault(rec_id, ([], []))
            r.extend(rels)
            l.extend(labels)
    own = state.get("recording_credits", {})
    dedupe = lambda xs: [json.loads(x) for x in dict.fromkeys(map(json.dumps, xs))]
    return {rid: (own.get(rid, []), *map(dedupe, shared.get(rid, ([], []))))
            for rid in state["recordings"]}


def role_names(own, shared, types):
    """Names credited with one of `types`: on the recording, else on its releases."""
    pick = lambda rels: sorted({name for t, _, name, _ in rels if t in types})
    return pick(own) or pick(shared)


# a fake release lookup: a producer on the release, a guitarist and a backing singer on
# the recording, one label; the recording is on two releases with the same producer
_rel = {"relations": [{"type": "producer", "artist": {"id": "p", "name": "Prod"}}],
        "label-info": [{"label": {"id": "l", "name": "Columbia"}}, {"label": None}]}
_rec = {"relations": [
    {"type": "instrument", "attributes": ["guest", "guitar"], "artist": {"id": "g", "name": "Gtr"}},
    {"type": "vocal", "attributes": ["background vocals"], "artist": {"id": "v", "name": "Voc"}}]}
_entry = [artist_rels(_rel), release_labels(_rel), ["r"]]
_own, _shared, _labels = song_credits({"recordings": {"r": []},
                                       "recording_credits": {"r": artist_rels(_rec)},
                                       "release_credits": {"x": _entry, "y": _entry}})["r"]
assert instruments_of(_own) == ["guitar"]
assert role_names(_own, _shared, ROLES["producers"]) == ["Prod"]  # falls back to release
assert role_names(_own, _shared, ROLES["vocals"]) == ["Voc"]
assert _shared == [["producer", "p", "Prod", []]] and _labels == [["l", "Columbia"]]  # deduped
del _rel, _rec, _entry, _own, _shared, _labels


def write_log(year, started, requests_before, completed):
    """Append this run's collection metadata to collection_log.jsonl."""
    state = load_state()
    credits = song_credits(state)
    with_credit = lambda test: sum(bool(test(*c)) for c in credits.values())
    entry = {
        "year": int(year), "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed": completed, "endpoint": API, "release_query": QUERY,
        "credits_inc": CREDITS_INC, "rate_limit_s_per_request": RATE_LIMIT_SECONDS,
        "requests_this_run": stats["requests"] - requests_before[0],
        "retried_requests_this_run": stats["retries"] - requests_before[1],
        "releases_found": len(state["releases"]),
        "releases_reported_by_api": state.get("search_total"),
        "recordings": len(state["recordings"]),
        "artists_with_genre_lookup": len(state.get("artists", {})),
        "releases_credit_lookup_done": len(state.get("release_credits", {})),
        "recordings_with_any_credit": with_credit(lambda o, s, l: o or s or l),
        **{f"recordings_with_{role}": with_credit(lambda o, s, l, t=types: role_names(o, s, t))
           for role, types in ROLES.items()},
        "recordings_with_labels": with_credit(lambda o, s, l: l),
    }
    COLLECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(COLLECTION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="search phase only")
    ap.add_argument("--no-credits", action="store_true",
                    help="skip the slow credits phase (the credit columns are left "
                         "empty for songs not looked up yet)")
    ap.add_argument("--years", nargs=2, type=int, default=[FIRST_YEAR, LAST_YEAR],
                    metavar=("FIRST", "LAST"), help="crawl every year FIRST..LAST")
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(args.years[0], args.years[1] + 1):
        print(f"\n##### {year} #####")
        set_year(year)
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        before, completed = (stats["requests"], stats["retries"]), False
        try:
            crawl_year(args.quick, not args.no_credits)
            completed = True
        finally:  # also log an interrupted run: its requests count toward the total
            write_log(YEAR, started, before, completed)


def crawl_year(quick, with_credits=True):
    state = load_state()
    if any(len(rec) < 7 for rec in state["recordings"].values()):
        # checkpoint from before genres/year/country were collected: redo the lookup phase
        print("Old checkpoint format, re-running phase 2 to collect year/country/language")
        state["recordings"], state["done_releases"] = {}, []
    print("Phase 1: searching releases...")
    phase1_search(state)
    releases = state["releases"]
    upper = sum(r["track_count"] or 0 for r in releases.values())
    print(f"\nReleases found: {len(releases)} (API reported {state['search_total']})")
    print(f"Upper bound (sum of track counts, not deduped): {upper}")

    if quick:
        return

    print("\nPhase 2: looking up tracklists...")
    phase2_lookup(state)
    print(f"\nDistinct English-language recordings first released in {YEAR}: "
          f"{len(state['recordings'])}")

    print("\nPhase 3: looking up artist genres...")
    phase3_genres(state)
    genre_names = fetch_genre_names()

    if with_credits:
        print("\nPhase 4: looking up credits (producers, musicians, singers, labels)...")
        phase4_credits(state)
    credits = song_credits(state)

    with open(musicbrainz_raw(YEAR), "w", encoding="utf-8") as f:
        f.write("id,song,artist,genre,year,language,country,genres,instruments,"
                + ",".join(ROLES) + ",labels\n")
        esc = lambda s: '"' + (s or "").replace('"', '""') + '"'
        for rid, rec in state["recordings"].items():
            genres = song_genres(rec, state["artists"], genre_names)
            own, shared, labels = credits[rid]
            people = [esc(";".join(role_names(own, shared, types))) for types in ROLES.values()]
            f.write(f"{rid},{esc(rec[0])},{esc(rec[1])},"
                    f"{esc(genres[0] if genres else '')},{rec[4]},{rec[6] or ''},"
                    f"{rec[5] or ''},{esc(';'.join(genres))},"
                    f"{esc(';'.join(instruments_of(own) or instruments_of(shared)))},"
                    f"{','.join(people)},{esc(';'.join(sorted({n for _, n in labels})))}\n")
    print(f"Wrote {musicbrainz_raw(YEAR)}")

    with open(musicbrainz_credits(YEAR), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording_id", "level", "type", "attributes", "mbid", "name"])
        for rid, (own, shared, labels) in credits.items():
            for level, rels in (("recording", own), ("release", shared)):
                for t, mbid, name, attrs in rels:
                    w.writerow([rid, level, t, ";".join(attrs), mbid, name])
            for mbid, name in labels:
                w.writerow([rid, "release", "label", "", mbid, name])
    print(f"Wrote {musicbrainz_credits(YEAR)}")


if __name__ == "__main__":
    main()