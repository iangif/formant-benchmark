"""Static figures summarising benchmark results across datasets, trackers, and formants."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from formant_benchmark.analysis.load import speaker_bootstrap_ci  # noqa: E402

# Colours are the first three slots of a CVD-validated categorical palette. The aqua
# slot is below 3:1 contrast on the surface, so trackers also differ by marker and
# line style, and every figure has a legend.
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3de"
_GOLD = "#8a8984"
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
_MARKERS = ("o", "s", "^")
_LINESTYLES = ("-", "--", ":")
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "benchmark_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

TRACKER_ORDER = ("praat_burg", "fasttrackpy", "formants_tracker")
TRACKER_LABELS = {"praat_burg": "Praat (Burg)", "fasttrackpy": "FastTrackPy", "formants_tracker": "FormantsTracker"}
DATASET_ORDER = ("hillenbrand", "vtr", "mcqll_english", "mcqll_french", "mcqll_japanese", "mcqll_uyghur")
DATASET_LABELS = {
    "hillenbrand": "Hillenbrand",
    "vtr": "VTR",
    "mcqll_english": "MCQLL\nEnglish",
    "mcqll_french": "MCQLL\nFrench",
    "mcqll_japanese": "MCQLL\nJapanese",
    "mcqll_uyghur": "MCQLL\nUyghur",
}
METRICS = {
    "mae": ("Mean absolute error", "Hz", "lower is better"),
    "rmse": ("Root mean square error", "Hz", "lower is better"),
    "fdr": ("Formant detection rate", "proportion", "higher is better"),
    "coverage": ("Coverage", "proportion", "higher is better"),
}

_STYLE = {
    "figure.facecolor": _SURFACE,
    "axes.facecolor": _SURFACE,
    "savefig.facecolor": _SURFACE,
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelcolor": _TEXT_SECONDARY,
    "axes.edgecolor": _GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": _GRID,
    "grid.linewidth": 0.6,
    "xtick.color": _TEXT_SECONDARY,
    "ytick.color": _TEXT_SECONDARY,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "text.color": _TEXT,
    "legend.frameon": False,
}


def plot_overview_heatmap(units: pd.DataFrame, metric: str = "mae") -> Figure:
    """Trackers x datasets grid of the mean unit metric, one panel per formant."""
    name, unit, direction = METRICS[metric]
    trackers, datasets, formants = _trackers(units), _datasets(units), _formants(units)
    means = units.groupby(["formant", "tracker", "dataset"])[metric].mean()
    rate = unit == "proportion"
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(
            len(formants), 1, figsize=(7.0, (0.42 * len(trackers) + 0.45) * len(formants) + 0.8),
            sharex=True, layout="constrained",
        )
        axes = np.atleast_1d(axes)
        for ax, formant in zip(axes, formants):
            grid = np.array([[means.get((formant, t, d), np.nan) for d in datasets] for t in trackers])
            # Rates share one 0-1 scale; Hz scales differ so much by formant that each panel gets its own.
            vmin, vmax = (0.0, 1.0) if rate else (0.0, np.nanmax(grid))
            image = ax.imshow(grid, cmap=_SEQUENTIAL, vmin=vmin, vmax=vmax, aspect="auto")
            for (row, col), value in np.ndenumerate(grid):
                if np.isfinite(value):
                    dark = (value - vmin) / (vmax - vmin) > 0.55
                    ax.text(
                        col, row, f"{value:.2f}" if rate else f"{value:.0f}",
                        ha="center", va="center", fontsize=8, color=_SURFACE if dark else _TEXT,
                    )
            ax.set_title(formant, loc="left")
            ax.set_xticks(range(len(datasets)), [DATASET_LABELS.get(d, d) for d in datasets])
            ax.tick_params(labelbottom=ax is axes[-1])
            ax.set_yticks(range(len(trackers)), [TRACKER_LABELS.get(t, t) for t in trackers])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            colorbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.015, aspect=10)
            colorbar.outline.set_visible(False)
            colorbar.ax.tick_params(colors=_TEXT_SECONDARY, size=0)
            if not rate:
                colorbar.set_label(unit)
        fig.suptitle(f"{name} by tracker and dataset ({direction})", x=0.01, ha="left", fontweight="bold")
    return fig


def plot_metric_dots(units: pd.DataFrame, metric: str = "mae", *, n_resamples: int = 1000) -> Figure:
    """Mean metric per formant with 95% CIs from resampling speakers, one panel per dataset."""
    name, unit, direction = METRICS[metric]
    trackers, datasets, formants = _trackers(units), _datasets(units), _formants(units)
    offsets = np.linspace(-0.22, 0.22, len(trackers)) if len(trackers) > 1 else np.zeros(1)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(
            len(datasets), 1, figsize=(6.5, 1.15 * len(datasets) + 0.9), sharex=True, layout="constrained"
        )
        axes = np.atleast_1d(axes)
        for ax, dataset in zip(axes, datasets):
            for position, tracker in enumerate(trackers):
                index = _slot(tracker)
                for row, formant in enumerate(formants):
                    subset = units.loc[
                        (units["dataset"] == dataset) & (units["tracker"] == tracker) & (units["formant"] == formant)
                    ]
                    if subset.empty:
                        continue
                    mean, low, high = speaker_bootstrap_ci(subset[metric], subset["speaker_id"], n_resamples=n_resamples)
                    y = row + offsets[position]
                    ax.plot([low, high], [y, y], color=_series(index), linewidth=1.5, solid_capstyle="round")
                    ax.plot(
                        mean, y, marker=_MARKERS[index % 3], markersize=5.5, color=_series(index),
                        markeredgecolor=_SURFACE, markeredgewidth=0.8, linestyle="none",
                    )
            ax.set_yticks(range(len(formants)), formants)
            ax.set_ylim(len(formants) - 0.5, -0.5)
            ax.grid(axis="y", visible=False)
            ax.set_title(_dataset_label(dataset), loc="left")
        axes[-1].set_xlabel(f"{name} ({unit}), 95% CI over speakers")
        axes[-1].set_xlim(left=0)
        fig.legend(handles=_tracker_handles(trackers, marker=True, dashed=False), loc="outside lower center", ncols=len(trackers))
        fig.suptitle(f"{name} per formant ({direction})", x=0.01, ha="left", fontweight="bold")
    return fig


def plot_error_ecdf(units: pd.DataFrame, metric: str = "mae") -> Figure:
    """Cumulative distribution of per-unit error: datasets in rows, formants in columns."""
    name, unit, _ = METRICS[metric]
    trackers, datasets, formants = _trackers(units), _datasets(units), _formants(units)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(
            len(datasets), len(formants), figsize=(2.6 * len(formants) + 0.6, 1.5 * len(datasets) + 0.9),
            sharex="col", sharey=True, layout="constrained", squeeze=False,
        )
        for col, formant in enumerate(formants):
            column_values = units.loc[units["formant"] == formant, metric].dropna()
            # The far tail would squash the informative part of every curve.
            upper = float(np.percentile(column_values, 98)) if len(column_values) else 1.0
            for row, dataset in enumerate(datasets):
                ax = axes[row, col]
                for tracker in trackers:
                    index = _slot(tracker)
                    values = units.loc[
                        (units["dataset"] == dataset) & (units["tracker"] == tracker) & (units["formant"] == formant),
                        metric,
                    ].dropna().to_numpy()
                    if not len(values):
                        continue
                    ordered = np.sort(values)
                    shares = np.arange(1, len(ordered) + 1) / len(ordered)
                    ax.step(ordered, shares, where="post", color=_series(index), linestyle=_LINESTYLES[index % 3], linewidth=1.5)
                ax.set_xlim(0, upper)
                ax.set_ylim(0, 1.02)
                if row == 0:
                    ax.set_title(formant)
                if col == 0:
                    ax.set_ylabel(_dataset_label(dataset), rotation=0, ha="right", va="center")
                if row == len(datasets) - 1:
                    ax.set_xlabel(f"{name} per unit ({unit})")
        fig.legend(handles=_tracker_handles(trackers, marker=False), loc="outside lower center", ncols=len(trackers))
        fig.suptitle(
            f"Share of units with error at or below each value (higher curve is better; x axis cut at 98th percentile)",
            x=0.01, ha="left", fontweight="bold",
        )
    return fig


def plot_position_profiles(positions: pd.DataFrame) -> Figure:
    """Mean absolute error by relative position in the vowel: datasets in rows, formants in columns."""
    trackers, datasets, formants = _trackers(positions), _datasets(positions), _formants(positions)
    means = positions.groupby(["dataset", "tracker", "formant", "position"])["abs_error"].mean()
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(
            len(datasets), len(formants), figsize=(2.6 * len(formants) + 0.6, 1.5 * len(datasets) + 0.9),
            sharex=True, layout="constrained", squeeze=False,
        )
        for row, dataset in enumerate(datasets):
            for col, formant in enumerate(formants):
                ax = axes[row, col]
                for tracker in trackers:
                    index = _slot(tracker)
                    try:
                        profile = means.loc[(dataset, tracker, formant)]
                    except KeyError:
                        continue
                    ax.plot(
                        profile.index * 100, profile.to_numpy(), color=_series(index), linestyle=_LINESTYLES[index % 3],
                        marker=_MARKERS[index % 3], markersize=3.5, linewidth=1.5,
                    )
                ax.set_ylim(bottom=0)
                ax.set_xlim(0, 100)
                if row == 0:
                    ax.set_title(formant)
                if col == 0:
                    ax.set_ylabel(_dataset_label(dataset), rotation=0, ha="right", va="center")
                if row == len(datasets) - 1:
                    ax.set_xlabel("Position in vowel (%)")
        fig.legend(handles=_tracker_handles(trackers, marker=True), loc="outside lower center", ncols=len(trackers))
        fig.suptitle(
            "Mean absolute error (Hz) by position in the vowel (Hillenbrand: source 10-80% points; others: 10% bins)",
            x=0.01, ha="left", fontweight="bold",
        )
    return fig


def plot_vowel_space(units: pd.DataFrame, dataset: str, *, min_units: int = 5) -> Figure:
    """F1/F2 vowel means: gold in grey with an arrow to each tracker's mean prediction.

    Means pool all speakers, so gender and speaker differences are averaged in.
    Vowels with fewer than *min_units* units are left out.
    """
    subset = units.loc[units["dataset"] == dataset]
    trackers = _trackers(subset)
    wide = subset.pivot_table(
        index=["tracker", "evaluation_unit_id", "vowel"], columns="formant", values=["gold_mean", "pred_mean"]
    ).dropna()
    means = wide.groupby(["tracker", "vowel"]).mean()
    counts = wide.groupby(["tracker", "vowel"]).size()
    means = means.loc[counts >= min_units]
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(
            1, len(trackers), figsize=(3.4 * len(trackers), 3.6), sharex=True, sharey=True, layout="constrained"
        )
        axes = np.atleast_1d(axes)
        for ax, tracker in zip(axes, trackers):
            index = _slot(tracker)
            if tracker not in means.index.get_level_values("tracker"):
                continue
            table = means.loc[tracker]
            for vowel, values in table.iterrows():
                gold = (values[("gold_mean", "F2")], values[("gold_mean", "F1")])
                predicted = (values[("pred_mean", "F2")], values[("pred_mean", "F1")])
                ax.annotate(
                    "", xy=predicted, xytext=gold,
                    arrowprops={"arrowstyle": "-|>", "color": _series(index), "linewidth": 1.2, "mutation_scale": 7},
                )
                ax.plot(*gold, marker="o", markersize=4, color=_GOLD, linestyle="none")
                # Annotations don't widen the axes, so the prediction is also plotted to keep arrows in view.
                ax.plot(*predicted, marker=_MARKERS[index % 3], markersize=3, color=_series(index), linestyle="none")
                ax.annotate(str(vowel), gold, xytext=(3, 3), textcoords="offset points", fontsize=7, color=_TEXT_SECONDARY)
            ax.set_title(TRACKER_LABELS.get(tracker, tracker), loc="left", color=_TEXT)
            ax.set_xlabel("F2 (Hz)")
        axes[0].set_ylabel("F1 (Hz)")
        # Phonetic convention: front vowels on the left, high vowels at the top.
        axes[0].invert_xaxis()
        axes[0].invert_yaxis()
        handles = [Line2D([], [], marker="o", color=_GOLD, linestyle="none", markersize=4, label="Gold mean")]
        handles += [
            Line2D([], [], color=_series(_slot(t)), linewidth=1.2, label=f"{TRACKER_LABELS.get(t, t)} mean")
            for t in trackers
        ]
        fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
        fig.suptitle(
            f"{_dataset_label(dataset)}: vowel means, gold to prediction", x=0.01, ha="left", fontweight="bold"
        )
    return fig


def _dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset).replace("\n", " ")


def _slot(tracker: str) -> int:
    """Fixed style slot per tracker, so a tracker keeps its colour in every figure and subset."""
    return TRACKER_ORDER.index(tracker) if tracker in TRACKER_ORDER else len(TRACKER_ORDER)


def _series(index: int) -> str:
    # A fourth tracker would need folding or faceting; the palette validates three.
    return _SERIES[index % len(_SERIES)]


def _tracker_handles(trackers: list[str], *, marker: bool, dashed: bool = True) -> list[Line2D]:
    return [
        Line2D(
            [], [], color=_series(_slot(t)), linewidth=1.5,
            marker=_MARKERS[_slot(t) % 3] if marker else None, linestyle=_LINESTYLES[_slot(t) % 3] if dashed else "-", markersize=5,
            label=TRACKER_LABELS.get(t, t),
        )
        for t in trackers
    ]


def _ordered(values, preferred: tuple[str, ...]) -> list[str]:
    present = set(map(str, values))
    return [value for value in preferred if value in present] + sorted(present - set(preferred))


def _trackers(frame: pd.DataFrame) -> list[str]:
    return _ordered(frame["tracker"].unique(), TRACKER_ORDER)


def _datasets(frame: pd.DataFrame) -> list[str]:
    return _ordered(frame["dataset"].unique(), DATASET_ORDER)


def _formants(frame: pd.DataFrame) -> list[str]:
    return sorted(map(str, frame["formant"].unique()))
