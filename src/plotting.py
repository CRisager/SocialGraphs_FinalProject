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
    """Save figures/<name>.pdf and figures/<name>.png. The PDF gets no creation date,
    so re-running the notebook only changes a PDF when the figure itself changes."""
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG_DIR / f"{name}.png")


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


def _null_hist(ax, null, real, color, symbol, what):
    """Histogram of null values with the real value as a vertical line, both labelled
    directly (no legend). The real value can sit inside the null distribution, so the
    two labels are on different rows: the null summary in the top-left corner, the real
    value one row lower, next to its line. Headroom above the bars keeps both clear."""
    null = np.asarray(null)
    counts, *_ = ax.hist(null, bins=15, color=color, alpha=0.55, linewidth=0)
    ax.axvline(real, color=color, linewidth=1.6)
    ax.set_ylim(0, counts.max() * 1.7)
    mean = f"{null.mean():.3f}".replace("-0.000", "0.000")  # no "-0.000" for tiny negatives
    ax.text(0.02, 0.98, f"{len(null)} {what}\n{mean} ± {null.std(ddof=1):.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK)
    lo, hi = ax.get_xlim()
    right_half = (real - lo) / (hi - lo) > 0.5   # label on the side with more room
    ax.annotate(f"real\n{symbol} = {real:.3f}", (real, 0.76),
                xycoords=("data", "axes fraction"), xytext=(-4 if right_half else 4, 0),
                textcoords="offset points", ha="right" if right_half else "left",
                va="top", fontsize=6)
    ax.grid(axis="x", visible=False)


def null_grid(nulls, reals, name=None, xlabels=None, symbol="M", what="null models"):
    """Rows of null histograms, one panel per graph: nulls = {row: {graph: array}},
    reals = {row: {graph: float}}, xlabels = {row: axis label}."""
    rows = list(nulls)
    ncol = max(len(v) for v in nulls.values())
    fig, axes = plt.subplots(len(rows), ncol, figsize=(PAGE_WIDTH, 1.9 * len(rows)),
                             squeeze=False)
    for i, row in enumerate(rows):
        for ax, (graph, null) in zip(axes[i], nulls[row].items()):
            _null_hist(ax, null, reals[row][graph], GRAPH_COLORS.get(graph, INK), symbol, what)
            ax.set_title(graph, loc="left")
            ax.set_xlabel((xlabels or {}).get(row, row))
        axes[i][0].set_ylabel(what)
    fig.tight_layout()
    if name:
        savefig(fig, name)
    return fig


def null_panels(nulls, reals, name=None, xlabel="modularity M"):
    """One histogram per graph of the null-model M values, with the real M as a
    vertical line ({graph: array}, {graph: float})."""
    return null_grid({xlabel: nulls}, {xlabel: reals}, name, symbol="M")


def null_matrix(nulls, reals, name=None, xlabels=None, whats=None, width=COL_WIDTH,
                panel_height=0.95):
    """All null-model checks in one figure for one PNAS column: one row per graph and
    one column per check. nulls = {check: {graph: array}}, reals = {check: {graph:
    float}}, xlabels / whats = {check: text}. The check is named once above its
    column and the graph once on the left of its row, so the panels carry no titles."""
    checks = list(nulls)
    graphs_ = list(next(iter(nulls.values())))
    fig, axes = plt.subplots(len(graphs_), len(checks), squeeze=False,
                             figsize=(width, 0.35 + panel_height * len(graphs_)))
    for j, check in enumerate(checks):
        for i, graph in enumerate(graphs_):
            ax = axes[i][j]
            null = np.asarray(nulls[check][graph])
            real = reals[check][graph]
            color = GRAPH_COLORS.get(graph, INK)
            counts, *_ = ax.hist(null, bins=12, color=color, alpha=0.55, linewidth=0)
            ax.axvline(real, color=color, linewidth=1.3)
            ax.set_ylim(0, counts.max() * 2.4)
            mean = f"{null.mean():.3f}".replace("-0.000", "0.000")
            ax.text(0.03, 0.97, f"null {mean}\n± {null.std(ddof=1):.3f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=5, color=INK)
            lo, hi = ax.get_xlim()
            right = (real - lo) / (hi - lo) > 0.5
            # bars reach at most 1/2.4 of the height, the null summary sits at the top:
            # the real-value label goes in the free band between them
            ax.annotate(f"real\n{real:.3f}", (real, 0.68), xycoords=("data", "axes fraction"),
                        xytext=(-3 if right else 3, 0), textcoords="offset points",
                        ha="right" if right else "left", va="top", fontsize=5)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(2))
            ax.tick_params(labelsize=5, pad=1.5)
            ax.grid(axis="x", visible=False)
            if i == len(graphs_) - 1:
                ax.set_xlabel((xlabels or {}).get(check, check), fontsize=6)
            if j == 0:
                ax.set_ylabel(f"{graph}\n{(whats or {}).get(check, 'count')}", fontsize=6)
            elif (whats or {}).get(check) != (whats or {}).get(checks[j - 1]):
                ax.set_ylabel((whats or {}).get(check, ""), fontsize=6)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    if name:
        savefig(fig, name)
    return fig


def heatmap_grid(tables, name=None, xlabels=None, width=COL_WIDTH, row_height=0.115,
                 vmax=100, cbar_label="% of community"):
    """Community x label heatmaps for one PNAS column: tables = {column: {graph:
    DataFrame of %}}, one row of panels per graph and one column per label type. Within
    a column every table has the same labels, so they are written only under the bottom
    panel; the graph is named once on the left of its row. Row heights follow the
    number of communities, so every cell has the same size."""
    cmap = mpl.colors.LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    cols = list(tables)
    graphs_ = list(next(iter(tables.values())))
    heights = [max(tables[c][g].shape[0] for c in cols) for g in graphs_]
    widths = [max(tables[c][g].shape[1] for g in graphs_) for c in cols]
    fig, axes = plt.subplots(len(graphs_), len(cols), squeeze=False, layout="constrained",
                             figsize=(width, 0.95 + row_height * sum(heights)),
                             gridspec_kw={"height_ratios": heights, "width_ratios": widths})
    for j, col in enumerate(cols):
        for i, graph in enumerate(graphs_):
            ax, t = axes[i][j], tables[col][graph]
            im = ax.imshow(t.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
            for (r, c), v in np.ndenumerate(t.to_numpy()):
                if v >= 0.5:
                    ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=4.3,
                            color="white" if v > 0.55 * vmax else "#0b0b0b")
            ax.set_yticks(range(t.shape[0]), t.index, fontsize=4.5)
            if i == len(graphs_) - 1:
                ax.set_xticks(range(t.shape[1]), t.columns, rotation=60, ha="right",
                              fontsize=5)
                ax.set_xlabel((xlabels or {}).get(col, col), fontsize=6)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(f"{graph}\ncommunity", fontsize=6)
            ax.grid(False)
            ax.tick_params(length=0, pad=1)
            for side in ax.spines.values():
                side.set_visible(False)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal",
                      fraction=0.025, aspect=40, pad=0.01)
    cb.set_label(cbar_label, fontsize=6)
    cb.ax.tick_params(labelsize=5)
    if name:
        savefig(fig, name)
    return fig


def wordcloud_grid(scores, name=None, color=INK, n_words=40, ncols=4, titles=None, seed=0):
    """One word cloud per community from {community: Series word -> score}; word size
    follows the score. All words have the graph's colour: colour carries no extra
    meaning here, so varying it would only suggest one."""
    from wordcloud import WordCloud
    keys = list(scores)
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_WIDTH, 1.45 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, c in zip(axes.flat, keys):
        freq = scores[c][scores[c] > 0].nlargest(n_words).to_dict()
        wc = WordCloud(width=640, height=400, background_color="white", prefer_horizontal=1.0,
                       color_func=lambda *a, **k: color, random_state=seed,
                       margin=4).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title((titles or {}).get(c, str(c)), loc="left", fontsize=6.5)
    fig.tight_layout(h_pad=1.6, w_pad=0.4)
    if name:
        savefig(fig, name)
    return fig


def box_panels(groups, name=None, ylabel="", colors=None, xlabels=None, height=2.0):
    """Box plots side by side: groups = {panel: {box label: values}}. Panel widths
    follow the number of boxes. A dashed line marks each panel's overall median, so
    boxes can be read as above or below it. Outliers are not drawn (the whiskers cover
    1.5 IQR); they would add ink but no comparison."""
    widths = [max(len(g), 3) for g in groups.values()]
    fig, axes = plt.subplots(1, len(groups), figsize=(PAGE_WIDTH, height), squeeze=False,
                             sharey=True, gridspec_kw={"width_ratios": widths})
    for ax, (panel, g) in zip(axes[0], groups.items()):
        color = (colors or {}).get(panel, GRAPH_COLORS.get(panel, INK))
        data = [np.asarray(v)[~np.isnan(np.asarray(v, dtype=float))] for v in g.values()]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops={"color": "#0b0b0b", "linewidth": 1.0},
                        whiskerprops={"color": INK, "linewidth": 0.6},
                        capprops={"color": INK, "linewidth": 0.6},
                        boxprops={"linewidth": 0})
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.6)
        ax.axhline(np.median(np.concatenate(data)), color=INK, linewidth=0.6, linestyle="--")
        ax.set_xticks(range(1, len(g) + 1), list(g), fontsize=6,
                      rotation=0 if max(len(str(k)) for k in g) <= 3 else 45,
                      ha="center" if max(len(str(k)) for k in g) <= 3 else "right")
        ax.set_title(panel, loc="left")
        ax.set_xlabel((xlabels or {}).get(panel, ""))
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel(ylabel)
    fig.tight_layout()
    if name:
        savefig(fig, name)
    return fig


# diverging blue <-> red with a neutral grey midpoint (reference palette): blue = less
# common than expected (lift < 1), red = more common (lift > 1)
DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#ef8a89", "#b8302f"]


def lift_heatmaps(lifts, n, name=None, min_count=5, lim=2.0, xlabels=None, column=False):
    """Lift tables ({panel: families x labels}) as heatmaps on a log2 scale from
    2^-lim to 2^lim. Cells with fewer than min_count songs (n, same shape) are left
    blank: a lift from 1-4 songs is mostly noise. The lift is printed in each cell.
    column=True stacks the panels for one PNAS column, with the colour bar below."""
    cmap = mpl.colors.LinearSegmentedColormap.from_list("div", DIVERGING)
    cmap.set_bad("white")
    rows = max(t.shape[0] for t in lifts.values())
    if column:
        fig, axes = plt.subplots(len(lifts), 1, figsize=(COL_WIDTH, 0.5 + 0.38 * rows),
                                 squeeze=False, layout="constrained")
        axes = axes.T
        cell_font, tick_font = 4.5, 5
    else:
        widths = [t.shape[1] for t in lifts.values()]
        fig, axes = plt.subplots(1, len(lifts), figsize=(PAGE_WIDTH, 0.9 + 0.17 * rows),
                                 squeeze=False, layout="constrained",
                                 gridspec_kw={"width_ratios": widths})
        cell_font, tick_font = 5, 6
    for ax, (panel, t) in zip(axes[0], lifts.items()):
        ok = (n[panel].reindex_like(t) >= min_count).to_numpy()
        with np.errstate(divide="ignore"):
            z = np.where(ok, np.log2(t.to_numpy(dtype=float)), np.nan)
        im = ax.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(t.to_numpy(dtype=float)):
            if ok[i, j]:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=cell_font,
                        color="white" if abs(z[i, j]) > 0.6 * lim else "#0b0b0b")
        ax.set_xticks(range(t.shape[1]), t.columns, rotation=60 if not column else 45,
                      ha="right", fontsize=tick_font)
        ax.set_yticks(range(t.shape[0]), t.index, fontsize=tick_font)
        ax.set_title(panel, loc="left", fontsize=6.5 if column else None)
        ax.set_xlabel((xlabels or {}).get(panel, ""), fontsize=6 if column else None)
        ax.grid(False)
        ax.tick_params(length=0, pad=1)
        for side in ax.spines.values():
            side.set_visible(False)
        if column:
            ax.set_ylabel("instrument family", fontsize=6)
    if not column:
        axes[0][0].set_ylabel("instrument family")
    ticks = np.arange(-lim, lim + 1)
    if column:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal",
                          fraction=0.03, aspect=40, pad=0.01, ticks=ticks)
        cb.set_label("lift (log scale)", fontsize=6)
        cb.ax.set_xticklabels([f"{2 ** t:g}" for t in ticks], fontsize=5)
    else:
        cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, pad=0.015, ticks=ticks,
                          label="lift (log scale)")
        cb.ax.set_yticklabels([f"{2 ** t:g}" for t in ticks])
    if name:
        savefig(fig, name)
    return fig
