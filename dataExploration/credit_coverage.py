"""
Credit coverage of one year's songs on MusicBrainz vs Discogs, per genre.

Genre is the MusicBrainz 5-label scheme mb_label (genre_label in dataCleaner/combine.py),
rap dropped. Per genre:
  songs, share     how many songs, and their share of the labelled songs
  matched          share found on the source (MusicBrainz: all, they come from there)
  instrument       share with >= 1 instrument credit. MusicBrainz falls back from the
                   recording to its releases; for Discogs both track-level and
                   track-or-album are shown
  producer         share with >= 1 producer credit (MusicBrainz: only once the crawler
                   has been re-run with the producers column)
  over-rep.        genre's share of instrument-credited songs / its share of songs
                   (1 = proportional, > 1 = over-represented)

Reads songs_<year>.csv (MusicBrainz crawler) and discogs_<year>.csv (Discogs crawler).
Writes dataExploration/credit_coverage_<year>.csv.

Usage:
    python dataExploration/credit_coverage.py [--year 1970]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataCleaner"))
from combine import genre_label  # noqa: E402

LABELS = ["rock", "rb", "pop", "country"]  # mb_label without rap


def has(col):
    return col.fillna("").astype(str).str.len() > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=1970)
    year = ap.parse_args().year

    mb = pd.read_csv(ROOT / f"songs_{year}.csv")
    mb["genre_label"] = [genre_label(g, p) for g, p in zip(mb.genres, mb.genre)]
    songs = pd.DataFrame({"id": mb.id, "genre": mb.genre_label,
                          "mb instrument": has(mb.instruments)})
    if "producers" in mb:
        songs["mb producer"] = has(mb.producers)

    dc_path = ROOT / f"discogs_{year}.csv"
    if dc_path.exists():
        dc = pd.read_csv(dc_path).set_index("id").reindex(songs.id)
        songs["discogs matched"] = has(dc["match"]).to_numpy()
        songs["discogs instrument (track)"] = has(dc.instruments).to_numpy()
        songs["discogs instrument (track or album)"] = \
            (has(dc.instruments) | has(dc.album_instruments)).to_numpy()
        songs["discogs producer (track or album)"] = \
            (has(dc.producers) | has(dc.album_producers)).to_numpy()
    else:
        print(f"No discogs_{year}.csv yet: MusicBrainz only.\n")

    songs = songs[songs.genre.isin(LABELS)]
    flags = [c for c in songs.columns if c not in ("id", "genre")]
    by = songs.groupby("genre")
    table = pd.DataFrame({"songs": by.size()}).reindex(LABELS)
    table.loc["all"] = len(songs)
    table["share"] = table.songs / len(songs)
    for col in flags:
        table[col] = by[col].mean().reindex(LABELS).tolist() + [songs[col].mean()]
        if "instrument" in col:
            credited = by[col].sum().reindex(LABELS)
            ratio = (credited / credited.sum()) / table.share[LABELS]
            table[col + " over-rep."] = ratio.tolist() + [1.0]

    pct = table.copy()
    for col in pct.columns[1:]:
        pct[col] = pct[col].map(lambda v: f"{v:.2f}" if col.endswith("over-rep.")
                                else f"{v:.0%}")
    print(f"Credit coverage of {len(songs)} labelled {year} songs (rates per genre)\n")
    print(pct.T.to_string())
    out = ROOT / "dataExploration" / f"credit_coverage_{year}.csv"
    table.to_csv(out)
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
