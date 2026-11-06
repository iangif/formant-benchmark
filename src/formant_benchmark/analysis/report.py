"""Write every benchmark figure and a summary table for a set of evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from formant_benchmark.analysis.load import load_unit_metrics, speaker_bootstrap_ci
from formant_benchmark.analysis.plots import (
    METRICS,
    plot_error_ecdf,
    plot_metric_dots,
    plot_overview_heatmap,
    plot_position_profiles,
    plot_vowel_space,
)
from formant_benchmark.analysis.positions import load_position_errors
from formant_benchmark.config.discovery import EVALUATIONS_ROOT, PREPARED_ROOT, RUNS_ROOT

REPORT_ROOT = Path("reports")


def write_report(
    *,
    evaluations_root: str | Path = EVALUATIONS_ROOT,
    prepared_root: str | Path = PREPARED_ROOT,
    runs_root: str | Path = RUNS_ROOT,
    output: str | Path = REPORT_ROOT,
    formats: Sequence[str] = ("png", "pdf"),
    include_positions: bool = True,
) -> list[Path]:
    """Render all figures into *output* and return the written paths.

    ``summary.csv`` holds the numbers behind the figures: the mean of each metric per
    dataset, tracker, and formant, with speaker-bootstrap 95% CIs for MAE.
    """
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    units = load_unit_metrics(evaluations_root)
    written: list[Path] = []

    figures = {
        "heatmap_mae": plot_overview_heatmap(units, "mae"),
        "heatmap_fdr": plot_overview_heatmap(units, "fdr"),
        "heatmap_coverage": plot_overview_heatmap(units, "coverage"),
        "dots_mae": plot_metric_dots(units, "mae"),
        "ecdf_mae": plot_error_ecdf(units, "mae"),
    }
    for dataset in sorted(units["dataset"].unique()):
        figures[f"vowel_space_{dataset}"] = plot_vowel_space(units, dataset)
    if include_positions:
        positions = load_position_errors(units, prepared_root=prepared_root, runs_root=runs_root)
        figures["position_mae"] = plot_position_profiles(positions)

    for name, figure in figures.items():
        for extension in formats:
            path = destination / f"{name}.{extension}"
            figure.savefig(path, dpi=200)
            written.append(path)
        plt.close(figure)

    summary_path = destination / "summary.csv"
    _summary(units).to_csv(summary_path, index=False)
    written.append(summary_path)
    return written


def _summary(units: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in METRICS if metric in units]
    rows = []
    for (dataset, tracker, formant), group in units.groupby(["dataset", "tracker", "formant"], sort=True):
        row = {"dataset": dataset, "tracker": tracker, "formant": formant, "n_units": len(group)}
        row.update({metric: group[metric].mean() for metric in metrics})
        _, row["mae_ci_low"], row["mae_ci_high"] = speaker_bootstrap_ci(group["mae"], group["speaker_id"])
        rows.append(row)
    return pd.DataFrame(rows)
