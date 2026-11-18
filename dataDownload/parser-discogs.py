"""
Match the MusicBrainz songs (songs_<year>.csv) against a Discogs data dump
(https://data.discogs.com, releases.xml.gz) and collect their credits. Same output as
crawler-discogs.py, without the API: one pass over the dump covers every year at once.

Matching
  A release track matches a song when the normalised titles are equal (lowercase,
  parentheses dropped, letters and digits only; see norm() in crawler-discogs.py) and
  one of the track's artists (else the release's) equals the song's artist or main
  artist ("The" ignored). Of the releases holding the song, the one kept is
    1. dated within a year of the song (else the closest year), then
    2. with the most credits for that track, then
    3. the lowest release id.
  Credits are read exactly as in crawler-discogs.py: track level (on the track, or on
  the release naming the track) and album level (the whole release).

Output (repo root), per year
  discogs_<year>.csv, discogs_credits_<year>.csv   as crawler-discogs.py
  collection_log.jsonl                            one line per year (source "discogs dump")

Takes ~30-60 min on a recent laptop (one core: gzip + XML). Needs lxml.

Usage:
    pip install lxml
    python dataDownload/parser-discogs.py ~/Desktop/discogs/discogs_20260101_releases.xml.gz
    python dataDownload/parser-discogs.py <dump> --years 1970 1980
"""

import argparse
import csv
import gzip
import importlib.util
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree

_spec = importlib.util.spec_from_file_location(
    "crawler_discogs", Path(__file__).with_name("crawler-discogs.py"))
crawler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crawler)
ROOT, norm, main_artist = crawler.ROOT, crawler.norm, crawler.main_artist

PROGRESS_EVERY = 500_000  # releases


def artist_key(name):
    """'The Kinks (2)' -> 'kinks'"""
    return re.sub(r"^the ", "", norm(re.sub(r" \(\d+\)$", "", name or "")))


def song_index(songs):
    """artist key -> normalised title -> [(song id, year, rule)]"""
    index = {}
    for s in songs:
        title = norm(s["song"])
        for rule, artist in (("artist + title", s["artist"]),
                             ("main artist", main_artist(s["artist"]))):
            hits = index.setdefault(artist_key(artist), {}).setdefault(title, [])
            if all(h[0] != s["id"] for h in hits):
                hits.append((s["id"], int(s["year"]), rule))
    return index


def credits_xml(parent):
    return [{"id": a.findtext("id"), "name": a.findtext("name", ""),
             "role": a.findtext("role", ""), "tracks": a.findtext("tracks", "")}
            for a in parent.iterfind("extraartists/artist")]


def release_dict(elem):
    """A <release> element in the shape of the API's /releases/{id} response."""
    year = (elem.findtext("released") or "")[:4]
    return {"title": elem.findtext("title", ""), "year": int(year) if year.isdigit() else 0,
            "extraartists": credits_xml(elem),
            "tracklist": [{"position": t.findtext("position", ""), "title": t.findtext("title", ""),
                           "extraartists": credits_xml(t)} for t in elem.iter("track")],
            "labels": [{"id": l.get("id"), "name": l.get("name", "")}
                       for l in elem.iterfind("labels/label")],
            "genres": [g.text for g in elem.iterfind("genres/genre")],
            "styles": [g.text for g in elem.iterfind("styles/style")]}


def scan_release(elem, index, best, releases):
    """Record every song this release holds, keeping the best release per song."""
    release_artists = {artist_key(n.text) for n in elem.iterfind("artists/artist/name")}
    tracks = list(elem.iter("track"))
    track_artists = [{artist_key(n.text) for n in t.iterfind("artists/artist/name")}
                     or release_artists for t in tracks]
    if not index.keys() & set().union(release_artists, *track_artists):
        return  # cheap filter: none of our artists on this release
    rel, rid = None, elem.get("id")
    for i, (t, artists) in enumerate(zip(tracks, track_artists)):
        title = norm(t.findtext("title", ""))
        for a in artists & index.keys():
            for song_id, year, rule in index[a].get(title, []):
                rel = rel or crawler.trim(release_dict(elem))
                dy = abs((rel["year"] or 9999) - year)
                rank = (0 if dy <= 1 else dy, -len(crawler.song_credits(rel, i)), int(rid))
                if song_id not in best or rank < best[song_id][0]:
                    best[song_id] = (rank, {"rule": rule, "release": rid, "track": i})
                    releases[rid] = rel


def scan(dump, index):
    best, releases, n, start = {}, {}, 0, time.time()
    with gzip.open(dump, "rb") as f:
        for _, elem in etree.iterparse(f, tag="release", huge_tree=True):
            scan_release(elem, index, best, releases)
            n += 1
            elem.clear(keep_tail=True)  # free memory: the dump is ~80 GB unzipped
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            if n % PROGRESS_EVERY == 0:
                print(f"  {n:,} releases, {time.time() - start:.0f}s, "
                      f"{len(best):,} songs matched")
    matches = {k: m for k, (_, m) in best.items()}
    return matches, {m["release"]: releases[m["release"]] for m in matches.values()}, n


# a fake dump: song a is on a 1975 reissue and on a 1970 single (kept); song b is only
# credited to its main artist on Discogs; the third release has none of our artists
_xml = b"""<releases>
<release id="7"><artists><artist><name>The Kinks</name></artist></artists><released>1975</released>
 <tracklist><track><position>1</position><title>Rats</title></track></tracklist></release>
<release id="9"><artists><artist><name>Kinks, The</name></artist></artists><released>1970-05-00</released>
 <extraartists><artist><id>6</id><name>Ray Davies</name><role>Producer</role><tracks></tracks></artist></extraartists>
 <tracklist><track><position>A</position><title>Rats (Mono)</title>
  <artists><artist><name>Kinks (2)</name></artist></artists>
  <extraartists><artist><id>5</id><name>Dave Davies</name><role>Guitar</role></artist></extraartists></track>
 <track><position>B</position><title>Got To Be Real</title>
  <artists><artist><name>Aretha Franklin</name></artist></artists></track></tracklist></release>
<release id="8"><artists><artist><name>Someone Else</name></artist></artists>
 <tracklist><track><title>Rats</title></track></tracklist></release>
</releases>"""
_idx = song_index([{"id": "a", "song": "Rats", "artist": "The Kinks", "year": "1970"},
                   {"id": "b", "song": "Got to Be Real", "year": "1970",
                    "artist": "Aretha Franklin with The Dixie Flyers"}])
_best, _rels = {}, {}
for _, _e in etree.iterparse(__import__("io").BytesIO(_xml), tag="release"):
    scan_release(_e, _idx, _best, _rels)
assert _best["a"][1] == {"rule": "artist + title", "release": "9", "track": 0}
assert _best["b"][1] == {"rule": "main artist", "release": "9", "track": 1}
assert crawler.names(crawler.song_credits(_rels["9"], 0), "musician", "track", 2) == ["guitar"]
del _xml, _idx, _best, _rels, _e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path, help="discogs_<date>_releases.xml.gz")
    ap.add_argument("--years", nargs=2, type=int, default=[1970, 1970],
                    metavar=("FIRST", "LAST"))
    args = ap.parse_args()
    by_year = {}
    for year in range(args.years[0], args.years[1] + 1):
        with open(ROOT / f"songs_{year}.csv", encoding="utf-8") as f:
            by_year[year] = list(csv.DictReader(f))
    songs = [s for ss in by_year.values() for s in ss]
    print(f"Scanning {args.dump.name} for {len(songs):,} songs ({args.years[0]}-{args.years[1]})")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    matches, releases, scanned = scan(args.dump, song_index(songs))
    state = {"songs": {s["id"]: matches.get(s["id"]) for s in songs}, "releases": releases}

    for year, year_songs in by_year.items():
        crawler.set_year(year)
        crawler.write_outputs(year_songs, state)
        entry = {"source": "discogs dump", "year": year, "started_utc": started,
                 "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "completed": True, "dump": args.dump.name, "releases_scanned": scanned,
                 "matching": "exact normalised title + artist or main artist; release "
                             "within a year of the song, then most track credits",
                 **crawler.coverage(year_songs, state)}
        with open(ROOT / "collection_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  {year}: {entry['songs_matched']}/{len(year_songs)} matched, "
              f"{entry['songs_with_any_instruments']} with instrument credits")


if __name__ == "__main__":
    main()
