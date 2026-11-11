"""Preprocess, explore, and cluster the 1970 song and lyric datasets.

Run after generating ``songs_1970.csv`` and filtering lyrics:

    python data_exploration.py --songs songs_1970.csv \
        --lyrics data/song_lyrics_1970.csv --output-dir analysis

The script writes cleaned data, feature matrices, plots, and community labels.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


def normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def clean_lyrics(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", str(text or ""))
    text = re.sub(r"[^A-Za-z0-9'\s]", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


def split_genres(value: str) -> list[str]:
    return sorted(
        {part.strip().casefold() for part in re.split(r"[;,|]", str(value or "")) if part.strip()}
    )


def load_songs(path: Path) -> pd.DataFrame:
    songs = pd.read_csv(path, dtype="string").fillna("")
    required = {"id", "song", "artist", "genre", "year", "language", "country", "genres"}
    missing = required - set(songs.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    songs["song_key"] = songs["song"].map(normalise)
    songs["artist_key"] = songs["artist"].map(normalise)
    songs["genre_list"] = songs["genres"].map(split_genres)
    songs["year"] = pd.to_numeric(songs["year"], errors="coerce").astype("Int64")
    songs = songs.drop_duplicates("id").drop_duplicates(["song_key", "artist_key"])
    return songs


def load_lyrics(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["song_key", "artist_key", "lyrics_clean"])
    lyrics = pd.read_csv(path, dtype="string", usecols=["title", "artist", "lyrics"]).fillna("")
    lyrics["song_key"] = lyrics["title"].map(normalise)
    lyrics["artist_key"] = lyrics["artist"].map(normalise)
    lyrics["lyrics_clean"] = lyrics["lyrics"].map(clean_lyrics)
    return lyrics.drop_duplicates(["song_key", "artist_key"])


def write_exploration_plots(songs: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for column, filename, title in [
        ("genre", "genre_counts.png", "Songs by primary genre"),
        ("country", "country_counts.png", "Songs by release country"),
        ("language", "language_counts.png", "Songs by language"),
    ]:
        counts = songs[column].replace("", "Unknown").value_counts().head(20).sort_values()
        ax = counts.plot.barh(figsize=(9, 6), title=title)
        ax.set_xlabel("Songs")
        ax.figure.tight_layout()
        ax.figure.savefig(output / filename, dpi=150)
        plt.close(ax.figure)

    lengths = songs["lyrics_clean"].str.split().str.len()
    if lengths.notna().any():
        ax = lengths[lengths > 0].plot.hist(bins=30, figsize=(9, 6), title="Lyrics length")
        ax.set_xlabel("Words")
        ax.figure.tight_layout()
        ax.figure.savefig(output / "lyrics_lengths.png", dpi=150)
        plt.close(ax.figure)


def build_features(songs: pd.DataFrame, output: Path) -> None:
    genres = MultiLabelBinarizer()
    genre_matrix = genres.fit_transform(songs["genre_list"])
    feature_names = [f"genre:{name}" for name in genres.classes_]
    feature_frame = pd.DataFrame(genre_matrix, columns=feature_names, index=songs.index)
    metadata = pd.get_dummies(
        songs[["language", "country"]].replace("", "Unknown"),
        prefix=["language", "country"],
        dtype=int,
    )
    feature_frame = pd.concat([feature_frame, metadata], axis=1)

    lyric_rows = songs["lyrics_clean"].fillna("")
    tfidf = TfidfVectorizer(
            min_df=2, max_df=0.95, stop_words="english", ngram_range=(1, 2)
        )
    if any(tfidf.build_analyzer()(text) for text in lyric_rows):
        lyric_matrix = tfidf.fit_transform(lyric_rows)
        lyric_frame = pd.DataFrame(
            lyric_matrix.toarray(),
            columns=[f"lyric:{term}" for term in tfidf.get_feature_names_out()],
            index=songs.index,
        )
        feature_frame = pd.concat([feature_frame, lyric_frame], axis=1)

    feature_frame.insert(0, "id", songs["id"].to_numpy())
    feature_frame.to_csv(output / "song_features.csv", index=False)

    if len(feature_frame) >= 2 and feature_frame.shape[1] > 1:
        n_clusters = max(2, min(8, len(feature_frame) // 10 or 2))
        matrix = feature_frame.drop(columns="id").to_numpy()
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(matrix)
        songs[["id", "song", "artist"]].assign(cluster=labels).to_csv(
            output / "song_clusters.csv", index=False
        )


def build_communities(songs: pd.DataFrame, output: Path) -> None:
    graph = nx.Graph()
    for _, row in songs.iterrows():
        song_id = str(row["id"])
        graph.add_node(song_id, song=row["song"], artist=row["artist"])
        entities = {f"genre:{genre}" for genre in row["genre_list"]}
        entities.update(f"artist:{artist.strip()}" for artist in str(row["artist"]).split(","))
        for entity in entities:
            graph.add_node(entity, bipartite="entity")
            graph.add_edge(song_id, entity)

    song_nodes = [node for node, data in graph.nodes(data=True) if "bipartite" not in data]
    projection = nx.bipartite.weighted_projected_graph(graph, song_nodes)
    projection.remove_edges_from(
        [(u, v) for u, v, data in projection.edges(data=True) if data["weight"] < 1]
    )
    communities = list(nx.community.greedy_modularity_communities(projection)) if projection else []
    labels = {
        song_id: index
        for index, community in enumerate(communities)
        for song_id in community
    }
    rows = [
        {"id": node, "community": labels.get(node, -1), "degree": degree}
        for node, degree in projection.degree()
    ]
    pd.DataFrame(rows).to_csv(output / "song_communities.csv", index=False)
    nx.write_gexf(projection, output / "song_similarity.gexf")
    print(
        f"Community graph: {projection.number_of_nodes()} songs, "
        f"{projection.number_of_edges()} edges, {len(communities)} communities"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", type=Path, default=Path("songs_1970.csv"))
    parser.add_argument("--lyrics", type=Path, default=Path("data/song_lyrics_1970.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    songs = load_songs(args.songs)
    lyrics = load_lyrics(args.lyrics)
    songs = songs.merge(lyrics[["song_key", "artist_key", "lyrics_clean"]], on=["song_key", "artist_key"], how="left")
    songs["lyrics_clean"] = songs["lyrics_clean"].fillna("")
    songs.to_csv(args.output_dir / "songs_clean.csv", index=False)
    summary = pd.DataFrame(
        {
            "rows": [len(songs)],
            "duplicate_ids": [songs["id"].duplicated().sum()],
            "missing_lyrics": [(songs["lyrics_clean"] == "").sum()],
            "unique_artists": [songs["artist_key"].nunique()],
            "unique_primary_genres": [songs["genre"].replace("", pd.NA).nunique()],
        }
    )
    summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))
    write_exploration_plots(songs, args.output_dir)
    build_features(songs, args.output_dir)
    build_communities(songs, args.output_dir)
    print(f"Processed {len(songs)} songs; lyrics matched for {(songs['lyrics_clean'] != '').sum()}.")


if __name__ == "__main__":
    main()
