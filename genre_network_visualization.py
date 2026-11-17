"""Create an interactive circular community view of the song genre network.

Songs are connected when their multi-label genre lists overlap.  The primary
``genre`` column controls node colors; the complete ``genres`` column controls
edges, so subgenre relationships are included as well.

Example:

    python genre_network_visualization.py

The script prefers ``songs_1970_lyrics.csv`` and ``songs_1970_clean.csv`` from
the cleaning pipeline, then falls back to ``Data/songs_1970.csv``.
"""

import argparse
import html
import json
import math
import random
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from data_exploration import load_songs


PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc949",
    "#af7aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]


def sampled_pairs(song_ids: list[str], limit: int, seed: int) -> list[tuple[str, str]]:
    """Sample pairwise combinations without materializing every combination."""
    count = len(song_ids)
    total = count * (count - 1) // 2
    amount = min(limit, total)
    if amount == 0:
        return []
    starts = [index * (2 * count - index - 1) // 2 for index in range(count - 1)]
    random_indices = random.Random(seed).sample(range(total), amount)
    pairs = []
    for pair_index in random_indices:
        left_index = bisect_right(starts, pair_index) - 1
        right_index = left_index + 1 + pair_index - starts[left_index]
        pairs.append((song_ids[left_index], song_ids[right_index]))
    return pairs


def build_genre_graph(
    songs: pd.DataFrame, max_edges_per_genre: int = 500
) -> nx.Graph:
    """Build a weighted song graph from sampled shared genre relationships."""
    graph = nx.Graph()
    graph.add_nodes_from(songs["id"].astype(str))
    genre_songs: dict[str, list[str]] = defaultdict(list)

    for _, row in songs.iterrows():
        song_id = str(row["id"])
        genres = set(row["genre_list"])
        if row["genre"]:
            genres.add(str(row["genre"]).casefold())
        for genre in genres:
            genre_songs[genre].append(song_id)

    shared_genres: dict[tuple[str, str], set[str]] = defaultdict(set)
    for genre_index, genre in enumerate(sorted(genre_songs)):
        for left, right in sampled_pairs(
            genre_songs[genre], max_edges_per_genre, seed=genre_index
        ):
            shared_genres[tuple(sorted((left, right)))].add(genre)

    for (left, right), genres in shared_genres.items():
        graph.add_edge(left, right, weight=len(genres), genres=sorted(genres))
    return graph


def circular_community_positions(
    graph: nx.Graph, communities: list[set[str]]
) -> dict[str, tuple[float, float]]:
    """Place each community on a proportional arc of a circle."""
    positions: dict[str, tuple[float, float]] = {}
    total_nodes = max(len(graph), 1)
    gap = math.radians(3)
    start = math.pi / 2

    for community in communities:
        span = 2 * math.pi * len(community) / total_nodes
        usable_span = max(span - gap, math.radians(1))
        nodes = sorted(community)
        for index, node in enumerate(nodes):
            fraction = (index + 0.5) / len(nodes)
            angle = start - gap / 2 - fraction * usable_span
            positions[node] = (math.cos(angle), math.sin(angle))
        start -= span
    return positions


def make_figure(
    songs: pd.DataFrame, max_edges_per_genre: int = 500
) -> tuple[go.Figure, dict[str, list[int]]]:
    graph = build_genre_graph(songs, max_edges_per_genre)
    communities = list(
        nx.community.greedy_modularity_communities(graph, weight="weight")
    )
    positions = circular_community_positions(graph, communities)

    primary_genres = songs["genre"].replace("", "Unknown").fillna("Unknown")
    genre_names = sorted(primary_genres.unique())
    colors = {
        genre: PALETTE[index % len(PALETTE)]
        for index, genre in enumerate(genre_names)
    }
    node_ids = songs["id"].astype(str).tolist()
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    adjacency: dict[str, list[int]] = defaultdict(list)
    for left, right in graph.edges:
        left_x, left_y = positions[left]
        right_x, right_y = positions[right]
        edge_x.extend([left_x, right_x, None])
        edge_y.extend([left_y, right_y, None])
        left_index = node_index[left]
        right_index = node_index[right]
        adjacency[left].append(right_index)
        adjacency[right].append(left_index)

    node_x = [positions[node_id][0] for node_id in node_ids]
    node_y = [positions[node_id][1] for node_id in node_ids]
    node_text = [
        (
            f"<b>{html.escape(str(row.song))}</b><br>"
            f"{html.escape(str(row.artist))}<br>"
            f"Year: {html.escape(str(row.year))}"
        )
        for row in songs.itertuples()
    ]
    node_colors = [colors[genre] for genre in primary_genres]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"color": "#56606b", "width": 0.6},
        opacity=0.25,
        hoverinfo="skip",
        showlegend=False,
    )
    highlight_trace = go.Scatter(
        x=[],
        y=[],
        mode="lines",
        line={"color": "#c4b5fd", "width": 2.4},
        hoverinfo="skip",
        showlegend=False,
    )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        text=node_text,
        customdata=node_ids,
        hovertemplate="%{text}<extra></extra>",
        marker={
            "size": 8,
            "color": node_colors,
            "line": {"color": "#111827", "width": 0.5},
        },
        showlegend=False,
    )

    figure = go.Figure(data=[edge_trace, highlight_trace, node_trace])
    for genre in genre_names:
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 9, "color": colors[genre]},
                name=genre,
                legendgroup="genres",
                showlegend=True,
            )
        )
    figure.update_layout(
        title="Songs connected by shared genres",
        template="plotly_dark",
        paper_bgcolor="#17191c",
        plot_bgcolor="#17191c",
        hoverlabel={"bgcolor": "#252a30", "font": {"size": 14}},
        legend={"title": "Overall genre", "itemsizing": "constant"},
        xaxis={"visible": False, "range": [-1.08, 1.08]},
        yaxis={
            "visible": False,
            "range": [-1.08, 1.08],
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    return figure, {
        str(node_index[node]): sorted(neighbors) for node, neighbors in adjacency.items()
    }


def write_visualization(
    songs: pd.DataFrame, output: Path, max_edges_per_genre: int = 500
) -> None:
    figure, adjacency = make_figure(songs, max_edges_per_genre)
    plot_id = "genre-network"
    adjacency_json = json.dumps(adjacency, separators=(",", ":"))
    post_script = f"""
const adjacency = {adjacency_json};
const plot = document.getElementById('{plot_id}');
const nodeTraceIndex = 2;
const highlightTraceIndex = 1;
function edgeCoordinates(nodeIndex) {{
    const coordinates = {{x: [], y: []}};
    (adjacency[nodeIndex] || []).forEach(function(neighborIndex) {{
        coordinates.x.push(plot.data[nodeTraceIndex].x[nodeIndex]);
        coordinates.x.push(plot.data[nodeTraceIndex].x[neighborIndex]);
        coordinates.x.push(null);
        coordinates.y.push(plot.data[nodeTraceIndex].y[nodeIndex]);
        coordinates.y.push(plot.data[nodeTraceIndex].y[neighborIndex]);
        coordinates.y.push(null);
    }});
    return coordinates;
}}
plot.on('plotly_hover', function(eventData) {{
    if (!eventData.points.length || eventData.points[0].curveNumber !== nodeTraceIndex) return;
    const nodeIndex = eventData.points[0].pointNumber;
    const coordinates = edgeCoordinates(nodeIndex);
    Plotly.restyle(plot, {{x: [coordinates.x], y: [coordinates.y]}}, [highlightTraceIndex]);
    const sizes = plot.data[nodeTraceIndex].x.map(function() {{ return 8; }});
    sizes[nodeIndex] = 13;
    Plotly.restyle(plot, {{'marker.size': [sizes]}}, [nodeTraceIndex]);
}});
plot.on('plotly_unhover', function() {{
    Plotly.restyle(plot, {{x: [[]], y: [[]]}}, [highlightTraceIndex]);
    Plotly.restyle(plot, {{'marker.size': [plot.data[nodeTraceIndex].x.map(function() {{ return 8; }})]}}, [nodeTraceIndex]);
}});
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        figure.to_html(
            full_html=True,
            include_plotlyjs=True,
            div_id=plot_id,
            post_script=post_script,
        ),
        encoding="utf-8",
    )


def default_songs_path() -> Path:
    """Prefer Jonathan's cleaned output, falling back to the original dataset."""
    for path in (
        Path("songs_1970-1980_lyrics.csv"),
        Path("songs_1970-1980_clean.csv"),
        Path("songs_1970_lyrics.csv"),
        Path("songs_1970_clean.csv"),
        Path("Data/songs_1970.csv"),
    ):
        if path.exists():
            return path
    return Path("songs_1970_lyrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--songs", type=Path, default=None)
    parser.add_argument(
        "--max-edges-per-genre",
        type=int,
        default=500,
        help="Maximum sampled edges contributed by each genre (default: 500)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/genre_network.html")
    )
    args = parser.parse_args()
    songs_path = args.songs or default_songs_path()
    songs = load_songs(songs_path)
    if songs.empty:
        raise ValueError(f"{songs_path} does not contain any songs")
    if args.max_edges_per_genre < 1:
        parser.error("--max-edges-per-genre must be at least 1")
    write_visualization(songs, args.output, args.max_edges_per_genre)
    graph = build_genre_graph(songs, args.max_edges_per_genre)
    print(
        f"Wrote {args.output}: {graph.number_of_nodes()} songs and "
        f"{graph.number_of_edges()} shared-genre edges from {songs_path}"
    )


if __name__ == "__main__":
    main()
