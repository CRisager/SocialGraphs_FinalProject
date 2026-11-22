"""
Fill in the credits (phase 4 of src/download/musicbrainz.py) from a MusicBrainz database
dump instead of ~1 API request per release (5+ hours for 1970-1980).

Written by Jonathan (branch `jonathan`, commit 8f67077, dataDownload/parser-musicBrainz.py);
moved into src/ with only the paths and imports changed. The credits in
data/raw/credits_<year>.csv and the instrument columns of songs_<year>.csv were made
with it (its collection-log entries match).

Phases 1-3 (which releases and songs, genres) are already in mb_<year>_checkpoint.json.
This script reads the relationship tables of the dump and writes, into each checkpoint,
exactly what phase 4 would have stored:
  recording_credits  recording id -> [type, artist id, artist name, attributes]
  release_credits    release id -> [its artist credits, its labels, our recordings on it]
Afterwards the crawler finds nothing left to fetch and only writes the CSVs.

Get the dump (https://data.metabrainz.org/pub/musicbrainz/data/fullexport/<date>/):
download mbdump.tar.bz2 only, and unpack just the tables used here (outside iCloud):
    tar -xjf mbdump.tar.bz2 mbdump/artist mbdump/label mbdump/link mbdump/link_type \\
        mbdump/link_attribute mbdump/link_attribute_type mbdump/l_artist_recording \\
        mbdump/l_artist_release mbdump/medium mbdump/recording mbdump/recording_gid_redirect \\
        mbdump/release mbdump/release_gid_redirect mbdump/release_label mbdump/track

Usage (repo root):
    python -m src.download.musicbrainz_dump ~/Desktop/musicbrainz/mbdump --years 1970 1980
    python -m src.download.musicbrainz --years 1970 1980   # writes the CSVs
"""

import argparse
import json
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import musicbrainz as crawler
from ..config import COLLECTION_LOG, musicbrainz_checkpoint

# Column positions in the dump's tab-separated tables (admin/sql/CreateTables.sql).
GID, NAME = 1, 2                 # artist, label, recording, release: id, gid, name, ...
MEDIUM_RELEASE = 1               # medium: id, release, ...
TRACK_RECORDING, TRACK_MEDIUM = 2, 3   # track: id, gid, recording, medium, ...
L_LINK, L_ARTIST, L_ENTITY = 1, 2, 3   # l_artist_*: id, link, entity0, entity1, ...
LINK_TYPE = 1                    # link: id, link_type, ...
LINK_TYPE_NAME = 6               # link_type: id, parent, child_order, gid, type0, type1, name
ATTR_NAME = 5                    # link_attribute_type: id, parent, root, child_order, gid, name
RL_RELEASE, RL_LABEL = 1, 2      # release_label: id, release, label, catalog_number


def rows(dump, table, keep=None, col=0):
    """Rows of a dump table, only those whose column `col` is in `keep` (if given)."""
    with open(Path(dump) / table, encoding="utf-8") as f:
        for line in f:
            r = line.rstrip("\n").split("\t")
            if keep is None or r[col] in keep:
                yield r


def text(s):
    """PostgreSQL COPY text: \\N is null, backslash escapes."""
    if s == r"\N":
        return None
    return re.sub(r"\\(.)", lambda m: {"t": "\t", "n": "\n", "r": "\r"}.get(m[1], m[1]), s)


def gid_to_id(dump, table, gids):
    """Our MusicBrainz ids -> internal row ids, following merges (gid redirects)."""
    ids = {r[GID]: r[0] for r in rows(dump, table, gids, GID)}
    ids.update({r[0]: r[1] for r in rows(dump, f"{table}_gid_redirect", gids - ids.keys())})
    return ids


def read_dump(dump, release_gids, recording_gids):
    """(recording credits, release credits) keyed by our ids, in the crawler's format."""
    rel_id = gid_to_id(dump, "release", release_gids)
    rec_id = gid_to_id(dump, "recording", recording_gids)
    our_recs = defaultdict(list)  # internal recording id -> our ids (merged ones share)
    for gid, i in rec_id.items():
        our_recs[i].append(gid)

    medium = {r[0]: r[MEDIUM_RELEASE] for r in rows(dump, "medium", set(rel_id.values()), MEDIUM_RELEASE)}
    on_release = defaultdict(set)
    for r in rows(dump, "track", medium.keys(), TRACK_MEDIUM):
        if r[TRACK_RECORDING] in our_recs:
            on_release[medium[r[TRACK_MEDIUM]]].update(our_recs[r[TRACK_RECORDING]])

    rels = {kind: defaultdict(list) for kind in ("recording", "release")}
    ids = {"recording": set(rec_id.values()), "release": set(rel_id.values())}
    for kind in rels:
        for r in rows(dump, f"l_artist_{kind}", ids[kind], L_ENTITY):
            rels[kind][r[L_ENTITY]].append((r[L_LINK], r[L_ARTIST]))
    found = [x for kind in rels for xs in rels[kind].values() for x in xs]
    links, artist_ids = {l for l, _ in found}, {a for _, a in found}

    type_name = {r[0]: r[LINK_TYPE_NAME] for r in rows(dump, "link_type")}
    attr_name = {r[0]: text(r[ATTR_NAME]) for r in rows(dump, "link_attribute_type")}
    link_type = {r[0]: type_name[r[LINK_TYPE]] for r in rows(dump, "link", links)}
    attrs = defaultdict(list)
    for r in rows(dump, "link_attribute", links):
        attrs[r[0]].append(attr_name[r[1]])
    artist = {r[0]: (r[GID], text(r[NAME])) for r in rows(dump, "artist", artist_ids)}
    assert {"instrument", "vocal", "producer"} <= set(type_name.values()), "link_type columns?"

    label_ids = defaultdict(list)
    for r in rows(dump, "release_label", ids["release"], RL_RELEASE):
        if r[RL_LABEL] != r"\N":
            label_ids[r[RL_RELEASE]].append(r[RL_LABEL])
    label = {r[0]: [r[GID], text(r[NAME])]
             for r in rows(dump, "label", {l for ls in label_ids.values() for l in ls})}

    credit = lambda l, a: [link_type[l], *artist[a], sorted(attrs[l])]
    rec_credits = {gid: [credit(l, a) for l, a in rels["recording"][i]]
                   for gid, i in rec_id.items()}
    rel_credits = {gid: [[credit(l, a) for l, a in rels["release"][i]],
                         [label[l] for l in label_ids[i]], sorted(on_release[i])]
                   for gid, i in rel_id.items()}
    return rec_credits, rel_credits


def _self_check():
    """A fake dump: release R (merged into r2) holds our recording A, played by G on
    guitar; R is produced by P and on label L. Recording B is not on R."""
    tables = {
        "release": "2\tr2\tTitle", "release_gid_redirect": "R\t2\t2020",
        "recording": "5\tA\tSong\n6\tB\tOther", "recording_gid_redirect": "",
        "medium": "9\t2\t1", "track": "1\tt1\t5\t9\t1\n2\tt2\t6\t8\t1",
        "l_artist_recording": "1\t30\t70\t5", "l_artist_release": "2\t31\t71\t2",
        "link": "30\t1\n31\t2", "link_type": "1\t\\N\t0\tg\tartist\trecording\tinstrument\n"
        "2\t\\N\t0\tg\tartist\trelease\tproducer\n3\t\\N\t0\tg\ta\tb\tvocal",
        "link_attribute": "30\t40\t2020\n30\t41\t2020",
        "link_attribute_type": "40\t\\N\t40\t0\tg\tguitar\n41\t\\N\t41\t0\tg\tguest",
        "artist": "70\tG\tGuitar Guy\n71\tP\tProd \\\\ Co", "release_label": "1\t2\t80\tX1\n2\t2\t\\N\t",
        "label": "80\tL\tColumbia",
    }
    with tempfile.TemporaryDirectory() as d:
        for name, body in tables.items():
            (Path(d) / name).write_text(body + "\n" if body else "")
        rec, rel = read_dump(d, {"R"}, {"A", "B"})
    assert rec == {"A": [["instrument", "G", "Guitar Guy", ["guest", "guitar"]]], "B": []}
    assert rel == {"R": [[["producer", "P", "Prod \\ Co", []]], [["L", "Columbia"]], ["A"]]}


_self_check()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path, help="the unpacked mbdump folder")
    ap.add_argument("--years", nargs=2, type=int, default=[1970, 1970],
                    metavar=("FIRST", "LAST"))
    args = ap.parse_args()
    years = range(args.years[0], args.years[1] + 1)
    paths = {y: musicbrainz_checkpoint(y) for y in years}
    states = {y: json.loads(p.read_text()) for y, p in paths.items()}
    releases = {gid for s in states.values() for gid in s["releases"]}
    recordings = {gid for s in states.values() for gid in s["recordings"]}
    print(f"Reading {args.dump} for {len(releases):,} releases, {len(recordings):,} recordings")
    rec_credits, rel_credits = read_dump(args.dump, releases, recordings)

    for year, state in states.items():
        state["recording_credits"] = {g: rec_credits.get(g, []) for g in state["recordings"]}
        # not in the dump (deleted): stored empty, as the crawler does for a 404
        state["release_credits"] = {g: rel_credits.get(g, [[], [], []]) for g in state["releases"]}
        tmp = paths[year].with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(paths[year])

        credits = crawler.song_credits(state)
        n_inst = sum(bool(crawler.instruments_of(o) or crawler.instruments_of(s))
                     for o, s, _ in credits.values())
        entry = {"source": "musicbrainz dump", "year": year, "dump": str(args.dump),
                 "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "releases": len(state["releases"]),
                 "releases_in_dump": sum(g in rel_credits for g in state["releases"]),
                 "recordings": len(state["recordings"]),
                 "recordings_in_dump": sum(g in rec_credits for g in state["recordings"]),
                 "recordings_with_instruments": n_inst}
        COLLECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(COLLECTION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        msg = f"  {year}: {n_inst}/{len(credits)} songs with instrument credits"
        if "instruments" in state:  # 1970 was crawled once via the API: compare
            old = {g: set(o or s) for g, (o, s) in state["instruments"].items()}
            new = {g: set(crawler.instruments_of(o) or crawler.instruments_of(s))
                   for g, (o, s, _) in credits.items()}
            msg += (f" (API crawl had {sum(map(bool, old.values()))}; "
                    f"same instruments for {sum(old[g] == new.get(g) for g in old)}/{len(old)})")
        print(msg)
    print("Now run: python -m src.download.musicbrainz "
          f"--years {args.years[0]} {args.years[1]}")


if __name__ == "__main__":
    main()
