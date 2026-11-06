"""Formant error as a function of relative position within the vowel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from formant_benchmark.config.discovery import PREPARED_ROOT, RUNS_ROOT
from formant_benchmark.data.io import load_prepared_dataset
from formant_benchmark.data.models import AnnotationType
from formant_benchmark.evaluation.align import interpolate_formant
from formant_benchmark.exceptions import EvaluationValidationError
from formant_benchmark.runs.io import load_prediction_run, read_prediction_run_manifest

_TIME_TOLERANCE_S = 1e-9


def load_position_errors(
    unit_metrics: pd.DataFrame,
    *,
    prepared_root: str | Path = PREPARED_ROOT,
    runs_root: str | Path = RUNS_ROOT,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return one row per gold observation with its relative vowel position and error.

    *unit_metrics* comes from :func:`load_unit_metrics` and decides which units and
    formants are included. Trajectory gold is re-aligned exactly as the evaluator
    does it (interpolated predictions at gold frame times), and each frame is put in
    one of *n_bins* equal bins of the vowel; ``position`` is the bin centre. Static
    gold keeps its source position (e.g. Hillenbrand's 10%-80% points) and needs no
    re-alignment. Columns: dataset, tracker, formant, speaker_id, evaluation_unit_id,
    position, error (prediction minus gold, Hz), abs_error.
    """
    frames: list[pd.DataFrame] = []
    for dataset_name, dataset_units in unit_metrics.groupby("dataset", sort=False):
        dataset = load_prepared_dataset(Path(prepared_root) / str(dataset_name))
        for _, units in dataset_units.groupby("evaluation", sort=False):
            if dataset.manifest.annotation_type is AnnotationType.STATIC:
                frame = _static_positions(dataset, units)
            else:
                run = load_prediction_run(_run_root(Path(runs_root), str(units["prediction_run_id"].iloc[0])))
                frame = _trajectory_positions(dataset, run.predictions, units, n_bins)
            frame.insert(0, "tracker", units["tracker"].iloc[0])
            frame.insert(0, "dataset", dataset_name)
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["abs_error"] = combined["error"].abs()
    return combined


def _static_positions(dataset, units: pd.DataFrame) -> pd.DataFrame:
    positions = dataset.static_measurements.set_index("measurement_id")["relative_position"]
    frame = units[["formant", "speaker_id", "evaluation_unit_id", "measurement_id"]].copy()
    frame["position"] = frame["measurement_id"].map(positions).to_numpy(dtype=float)
    frame["error"] = units["pred_mean"].to_numpy() - units["gold_mean"].to_numpy()
    # Steady-state points have a time but no relative position, so they drop out here.
    frame = frame.loc[np.isfinite(frame["position"]) & np.isfinite(frame["error"])]
    return frame.drop(columns="measurement_id")


def _trajectory_positions(dataset, predictions: pd.DataFrame, units: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    tracks = {str(item_id): frame for item_id, frame in dataset.tracks.groupby("item_id", sort=False)}
    predicted = {
        str(item_id): frame.sort_values("time_s", kind="stable")
        for item_id, frame in predictions.groupby("item_id", sort=False)
    }
    formants = list(units["formant"].unique())
    whole = (
        units[["evaluation_unit_id", "item_id", "interval_id", "speaker_id"]]
        .drop_duplicates("evaluation_unit_id")
        .merge(dataset.intervals[["interval_id", "start_s", "end_s"]], on="interval_id", how="left")
    )

    # Gather every gold frame inside an evaluated vowel, then interpolate once per item and formant.
    columns: dict[str, list[np.ndarray]] = {"formant": [], "speaker_id": [], "evaluation_unit_id": [], "position": [], "error": []}
    for item_id, item_units in whole.groupby("item_id", sort=False):
        gold_track = tracks.get(str(item_id))
        prediction = predicted.get(str(item_id))
        if gold_track is None or prediction is None:
            continue
        times = gold_track["time_s"].to_numpy(dtype=float)
        frame_index: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        unit_ids: list[np.ndarray] = []
        speakers: list[np.ndarray] = []
        for unit_id, speaker, start_s, end_s in zip(
            item_units["evaluation_unit_id"], item_units["speaker_id"], item_units["start_s"], item_units["end_s"]
        ):
            inside = np.flatnonzero((times >= start_s - _TIME_TOLERANCE_S) & (times <= end_s + _TIME_TOLERANCE_S))
            relative = np.clip((times[inside] - start_s) / (end_s - start_s), 0.0, 1.0)
            bins = np.minimum((relative * n_bins).astype(int), n_bins - 1)
            frame_index.append(inside)
            positions.append((bins + 0.5) / n_bins)
            unit_ids.append(np.full(len(inside), unit_id, dtype=object))
            speakers.append(np.full(len(inside), speaker, dtype=object))
        index = np.concatenate(frame_index)
        for formant in formants:
            error = interpolate_formant(prediction, times[index], formant) - gold_track[formant].to_numpy(dtype=float)[index]
            keep = np.isfinite(error)
            columns["formant"].append(np.full(int(keep.sum()), formant, dtype=object))
            columns["speaker_id"].append(np.concatenate(speakers)[keep])
            columns["evaluation_unit_id"].append(np.concatenate(unit_ids)[keep])
            columns["position"].append(np.concatenate(positions)[keep])
            columns["error"].append(error[keep])
    return pd.DataFrame({name: np.concatenate(parts) for name, parts in columns.items()})


def _run_root(runs_root: Path, run_id: str) -> Path:
    if (runs_root / run_id / "run_manifest.yaml").is_file():
        return runs_root / run_id
    for path in sorted(runs_root.iterdir()):
        if (path / "run_manifest.yaml").is_file() and read_prediction_run_manifest(path).run_id == run_id:
            return path
    raise EvaluationValidationError(f"No prediction run with id '{run_id}' under '{runs_root}'.")
