"""
Shared figure style and helpers for notebooks/analysis.ipynb.

Figures are sized for PNAS: one column is 8.7 cm (3.42 in), two columns 17.8 cm (7 in),
and text should be 6-8 pt at print size. Every figure is saved as PDF (vector, for the
report) and PNG (300 dpi, for slides and the notebook).

Each graph keeps one colour in every figure (GRAPH_COLORS: the first three slots of
the validated reference palette in src/config.py); when graphs are compared they get
one panel each, so colour is never the only way to tell them apart.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .config import FIG_DIR, PLOT_STYLE

COL_WIDTH = 3.42   # inches, one PNAS column
PAGE_WIDTH = 7.0   # inches, two columns
GRAPH_COLORS = {"G_lyrics": "#2a78d6", "G_instruments": "#eb6834", "G_combined": "#1baf7a"}
INK = "#52514e"

PNAS_STYLE = {
    **PLOT_STYLE,
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "lines.linewidth": 1.2, "grid.linewidth": 0.4,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # embed TrueType, so text stays editable in the PDF
}


def use_style():
    mpl.rcParams.update(PNAS_STYLE)


def savefig(fig, name):
    """Save figures/<name>.pdf and figures/<name>.png."""
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")


def ccdf(values):
    """(x, P(X >= x)) for the distinct values, for a log-log CCDF plot. The CCDF needs
    no binning, unlike a histogram, so the heavy tail is not hidden by bin choices."""
    x = np.sort(np.asarray(values, dtype=float))
    uniq, first = np.unique(x, return_index=True)
    return uniq, 1.0 - first / len(x)


assert np.allclose(ccdf([1, 1, 2, 3])[1], [1.0, 0.5, 0.25])


def ccdf_panels(series, xlabel, name=None, ylabel="P(X ≥ x)"):
    """One log-log CCDF panel per graph ({graph name: values}), in one row."""
    fig, axes = plt.subplots(1, len(series), figsize=(PAGE_WIDTH, 2.0), squeeze=False)
    for ax, (graph, values) in zip(axes[0], series.items()):
        x, y = ccdf(values)
        ax.loglog(x, y, color=GRAPH_COLORS.get(graph, INK), linewidth=1.2,
                  drawstyle="steps-post")
        ax.set_title(graph, loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="major")
        # a range under one decade makes matplotlib label the minor ticks too ("2x10^0"),
        # which overlap at column width: label only powers of ten
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_formatter(mpl.ticker.NullFormatter())
    axes[0][0].set_ylabel(ylabel)
    fig.tight_layout()
    if name:
        savefig(fig, name)
    return fig


def size_panels(sizes, name=None, ylabel="songs"):
    """One bar panel per graph ({graph name: community sizes}), communities ranked by
    size. The panels share no y-axis: the graphs have very different sizes."""
    fig, axes = plt.subplots(1, len(sizes), figsize=(PAGE_WIDTH, 1.9), squeeze=False)
    for ax, (graph, s) in zip(axes[0], sizes.items()):
        s = sorted(s, reverse=True)
        ax.bar(np.arange(len(s)), s, width=0.8, color=GRAPH_COLORS.get(graph, INK),
               linewidth=0)
        ax.set_title(f"{graph} ({len(s)} communities)", loc="left")
        ax.set_xlabel("community (by size)")
        ax.set_xticks(np.arange(len(s)), [str(i) for i in range(len(s))], fontsize=5.5)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel(ylabel)
    fig.tight_layout()
    if name:
        savefig(fig, name)
    return fig


# sequential blue ramp of the reference palette (light = near zero), for heatmaps
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
             "#0d366b"]


def heatmap_panels(tables, name=None, xlabel="", vmax=100):
    """Row-normalised community x label tables ({graph: DataFrame of %}) as heatmap
    panels side by side, with the percentage printed in each cell. One shared colour
    scale (0-vmax %), so the panels can be compared."""
    cmap = mpl.colors.LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    widths = [max(t.shape[1], 3) for t in tables.values()]
    height = 0.9 + 0.17 * max(t.shape[0] for t in tables.values())
    fig, axes = plt.subplots(1, len(tables), figsize=(PAGE_WIDTH, height), squeeze=False,
                             layout="constrained",
                             gridspec_kw={"width_ratios": widths})
    for ax, (graph, t) in zip(axes[0], tables.items()):
        im = ax.imshow(t.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(t.to_numpy()):
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if v > 0.55 * vmax else "#0b0b0b")
        ax.set_xticks(range(t.shape[1]), t.columns, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(t.shape[0]), t.index, fontsize=6)
        ax.set_title(graph, loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(False)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0][0].set_ylabel("community")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.015, label="% of community")
    if name:
        savefig(fig, name)
    return fig


def null_panels(nulls, reals, name=None, xlabel="modularity M"):
    """One histogram per graph of the null-model values, with the real value as a
    vertical line ({graph: array}, {graph: float})."""
    fig, axes = plt.subplots(1, len(nulls), figsize=(PAGE_WIDTH, 1.9), squeeze=False)
    for ax, (graph, null) in zip(axes[0], nulls.items()):
        color = GRAPH_COLORS.get(graph, INK)
        null = np.asarray(null)
        ax.hist(null, bins=15, color=color, alpha=0.55, linewidth=0)
        ax.axvline(reals[graph], color=color, linewidth=1.6)
        # direct labels instead of a legend: the real line, and the null distribution
        ax.annotate(f"real\nM = {reals[graph]:.3f}", (reals[graph], 0.97),
                    xycoords=("data", "axes fraction"), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="top", fontsize=6)
        ax.annotate(f"{len(null)} null models\n{null.mean():.3f} ± {null.std(ddof=1):.3f}",
                    (null.max(), 0.97), xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="top", fontsize=6, color=INK)
        ax.set_title(graph, loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("null models")
    fig.tight_layout()
    if name:
        savefig(fig, name)
    return fig
