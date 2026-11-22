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
