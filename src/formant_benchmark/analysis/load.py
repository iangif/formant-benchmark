"""Collect persisted evaluations into tidy, long-format tables for analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from formant_benchmark.config.discovery import EVALUATIONS_ROOT
from formant_benchmark.data.models import Formant
from formant_benchmark.evaluation.io import UNIT_VALUE_COLUMNS, load_evaluation_result
from formant_benchmark.exceptions import EvaluationValidationError

# F4 gold is unreliable in some datasets (e.g. VTR's is uncorrected), so analysis defaults to F1-F3.
DEFAULT_FORMANTS = ("F1", "F2", "F3")
_VALUE_NAMES = ("gold_mean", "pred_mean")


def load_unit_metrics(
    evaluations_root: str | Path = EVALUATIONS_ROOT,
    *,
    formants: tuple[str, ...] = DEFAULT_FORMANTS,
) -> pd.DataFrame:
    """Return one row per (evaluation, whole evaluation unit, formant) across all evaluations.

    Each row carries ``dataset``, ``tracker``, ``evaluation``, ``prediction_run_id``, ``formant``, the
    per-formant metrics and values without their formant suffix (``mae``,
    ``gold_mean``, ...), and the unit's identity and metadata columns. Central-region
    units are left out so every unit counts once.
    """
    root = Path(evaluations_root)
    paths = sorted(path for path in root.iterdir() if (path / "evaluation_manifest.yaml").is_file())
    if not paths:
        raise EvaluationValidationError(f"No evaluations found under '{root}'.")
    frames = [_long_units(path, formants) for path in paths]
    return pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)


def _long_units(path: Path, formants: tuple[str, ...]) -> pd.DataFrame:
    result = load_evaluation_result(path)
    manifest = result.manifest
    if manifest.schema_version == "1":
        raise EvaluationValidationError(
            f"'{path}' predates the gold_mean/pred_mean columns; re-run its experiment with --overwrite."
        )
    units = result.evaluation_unit_metrics
    units = units.loc[units["region"].isna() | (units["region"] == "full")]
    names = [*manifest.metrics, *_VALUE_NAMES]
    suffixed = {f"{name}_{formant.value.lower()}" for name in names for formant in Formant}
    shared = [column for column in units.columns if column not in {*suffixed, *manifest.metrics, *UNIT_VALUE_COLUMNS}]
    selected = {formant.value for formant in manifest.selected_formants}

    frames: list[pd.DataFrame] = []
    for formant in formants:
        if formant not in selected:
            continue
        frame = units[shared].copy()
        for name in names:
            frame[name] = units[f"{name}_{formant.lower()}"].to_numpy()
        frame.insert(0, "formant", formant)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "prediction_run_id", manifest.prediction_run_id)
    combined.insert(0, "evaluation", manifest.evaluation_id)
    combined.insert(0, "tracker", manifest.tracker)
    combined.insert(0, "dataset", manifest.dataset_name)
    return combined


def speaker_bootstrap_ci(
    values: pd.Series,
    speakers: pd.Series,
    *,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, low, high): the unit mean and a percentile CI from resampling speakers.

    Units from one speaker are not independent, so whole speakers are resampled with
    replacement and the unit mean is recomputed each time.
    """
    frame = pd.DataFrame({"value": values.to_numpy(dtype=float), "speaker": speakers.to_numpy()})
    frame = frame.loc[np.isfinite(frame["value"])]
    if frame.empty:
        return float("nan"), float("nan"), float("nan")
    grouped = frame.groupby("speaker")["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(grouped), size=(n_resamples, len(grouped)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(sums.sum() / counts.sum()), float(low), float(high)
