"""
Rough count of how many songs survive each filtering step, from the files the pipeline
has already written (nothing is downloaded or re-cleaned).

Usage:
    python dataExploration/filter_counts.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataCleaner"))
from clean import FIRST_YEAR, LAST_YEAR, SECTION_HEADER, YEARS  # noqa: E402

YEAR_RANGE = range(FIRST_YEAR, LAST_YEAR + 1)
MAX_SONGS = 8000  # same cap as clustering/lyrics_network.py


def musicbrainz():
    rows = {}
    for y in YEAR_RANGE:
        state = json.loads((ROOT / f"mb_{y}_checkpoint.json").read_text())
        rel = state["releases"].values()
        rows[y] = {"releases reported by API": state["search_total"],
                   "releases fetched": len(rel),
                   "tracks on them (upper bound)": sum(r["track_count"] or 0 for r in rel),
                   "recordings first released that year": len(pd.read_csv(ROOT / f"songs_{y}.csv"))}
    t = pd.DataFrame(rows).T
    s = pd.read_csv(ROOT / f"songs_{YEARS}_lyrics.csv")
    both = s.genius_tag.notna() & s.mb_label.notna()
    for name, mask in [("songs after merging duplicates", s.year.notna()),
                       ("with any MusicBrainz genre", s.genre.notna()),
                       ("genre maps to a Genius tag", s.mb_label.notna()),
                       ("with Genius lyrics", s.match.notna()),
                       ("lyrics + both labels", both),
                       ("both labels agree", s.labels_agree == True)]:  # noqa: E712
        t[name] = s[mask].groupby("year").size().reindex(t.index, fill_value=0)
    t.loc["total"] = t.sum()
    print("MusicBrainz side, per year\n" + t.T.to_string())
    print("\nHow the lyrics were matched\n" + s.match.value_counts().to_string())
    print("\nLabel used where both agree\n" + s[s.labels_agree == True].label.value_counts().to_string())  # noqa: E712
    print("\nGenius tag vs MusicBrainz label (songs with both)\n"
          + pd.crosstab(s.genius_tag[both], s.mb_label[both]).to_string())
    return t


def genius():
    g = pd.read_csv(ROOT / "genius_songs_trimmed.csv")
    g = g[g.year.between(FIRST_YEAR, LAST_YEAR)]
    steps = [("Genius pages in the year range", g)]
    g = g[g.tag != "misc"];                                   steps.append(("not tagged 'misc'", g))
    g = g[g.language.eq("en") | g.language.isna()];           steps.append(("English (or language unknown)", g))
    g = g[g.lyrics.str.replace(SECTION_HEADER, "", regex=True).str.strip() != ""]
    steps.append(("lyrics not empty", g))
    c = pd.read_csv(ROOT / f"genius_{YEARS}_clean.csv");      steps.append(("after merging versions/duplicates", c))
    c = c[c.tag != "rap"];                                    steps.append(("without rap (analysis scripts)", c))
    steps.append(("without pop (second analysis)", c[c.tag != "pop"]))
    t = pd.DataFrame({name: d.groupby("year").size() for name, d in steps}).fillna(0).astype(int)
    t.loc["total"] = t.sum()
    print("\n\nGenius side, per year\n" + t.T.to_string())
    print("\nGenius tags after cleaning\n" + steps[4][1].tag.value_counts().to_string())
    print(f"\nlanguage unknown but kept: {steps[2][1].language.isna().sum()}")
    for name, d in steps[5:]:
        print(f"clustering uses {min(len(d), MAX_SONGS)} of {len(d)} songs for '{name}' "
              f"(cap {MAX_SONGS})")
    return t


if __name__ == "__main__":
    mb, ge = musicbrainz(), genius()
    # each step can only remove songs
    order = ["recordings first released that year", "songs after merging duplicates",
             "with Genius lyrics", "lyrics + both labels", "both labels agree"]
    assert mb.loc["total", order].is_monotonic_decreasing
    assert ge.loc["total"].is_monotonic_decreasing
