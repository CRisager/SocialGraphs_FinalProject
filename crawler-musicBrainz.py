"""
Count English-language songs first released in 1975, using the MusicBrainz API.

Definition used (adjust to taste):
  * "release"  = an official release dated 1975 whose tracklist language is English (lang:eng)
  * "song"     = a distinct MusicBrainz *recording* whose first-release-date is in 1975
                 (so reissues of earlier tracks on 1975 compilations are not counted)
  * all genres  - no genre filter is applied, but each song is labelled with genres:
                 the recording's own genre tags if it has any, otherwise its artists'.
                 Only tags on MusicBrainz's official genre list count (so not "70s").

Three phases:
  1. Search  - pages through /ws/2/release search (100 per request) to list candidate releases.
               This alone gives a quick upper bound (sum of track counts).
  2. Lookup  - fetches the recordings of ~45 releases per request via the recording search
               and dedupes them. MusicBrainz allows ~1 request/second; progress is
               checkpointed, so you can stop (Ctrl+C) and re-run to resume.
  3. Genres  - fetches the tags of ~45 artists per request via the artist search.

Usage:
    pip install requests
    python crawler-musicBrainz.py            # both phases
    python crawler-musicBrainz.py --quick    # phase 1 only (fast upper bound)
"""

import argparse
import json
import os
import sys
import time

import requests

API = "https://musicbrainz.org/ws/2"
# MusicBrainz requires a descriptive User-Agent with contact info, or it will block you.
HEADERS = {"User-Agent": "SongsCounter/1.0 ( your-email@example.com )"}
RATE_LIMIT_SECONDS = 1.1
YEAR = "1970"
LANG = "eng"  # ISO 639-3 code of the tracklist language
CHECKPOINT = f"mb_{YEAR}_checkpoint.json"

# Lucene query for the release search. Drop the NOT clause to include compilations.
QUERY = f'date:{YEAR} AND lang:{LANG} AND status:official AND type:single AND NOT secondarytype:compilation'

session = requests.Session()
session.headers.update(HEADERS)
_last_call = 0.0


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
            "done_releases": [], "recordings": {}, "artists": {}}


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
SPLITTERS = [
    ("country", ["US", "GB", "CA", "AU", "NZ", "IE", "XE", "XW", "DE", "NL",
                 "FR", "JP", "SE", "IT", "ES", "BR", "ZA", "JM", "NO", "DK"]),
    ("primarytype", ["album", "single", "ep", "broadcast", "other"]),
    ("tracks", [str(n) for n in range(1, 41)]),
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
    state.setdefault("done_queries", [])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="search phase only")
    args = ap.parse_args()

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

    if args.quick:
        return

    print("\nPhase 2: looking up tracklists...")
    phase2_lookup(state)
    print(f"\nDistinct English-language recordings first released in {YEAR}: "
          f"{len(state['recordings'])}")

    print("\nPhase 3: looking up artist genres...")
    phase3_genres(state)
    genre_names = fetch_genre_names()

    with open(f"songs_{YEAR}.csv", "w", encoding="utf-8") as f:
        f.write("id,song,artist,genre,year,language,country,genres\n")
        esc = lambda s: '"' + (s or "").replace('"', '""') + '"'
        for rid, rec in state["recordings"].items():
            genres = song_genres(rec, state["artists"], genre_names)
            f.write(f"{rid},{esc(rec[0])},{esc(rec[1])},"
                    f"{esc(genres[0] if genres else '')},{rec[4]},{rec[6] or ''},"
                    f"{rec[5] or ''},{esc(';'.join(genres))}\n")
    print(f"Wrote songs_{YEAR}.csv")


if __name__ == "__main__":
    main()