"""Structural validation for normalized prepared datasets; no acoustic cleaning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from formant_benchmark.data.models import AnnotationType, Formant, IntervalOrigin, PreparedDataset
from formant_benchmark.data.schemas import (
    FORMANT_COLUMNS,
    INTERVAL_COLUMNS,
    ITEM_REQUIRED_COLUMNS,
    SPLIT_COLUMNS,
    STATIC_MEASUREMENT_COLUMNS,
    TRACK_COLUMNS,
)
from formant_benchmark.data.splits import validate_speaker_disjoint_splits
from formant_benchmark.exceptions import DatasetValidationError

_TIME_TOLERANCE_S = 1e-6


def validate_prepared_dataset(dataset: PreparedDataset, *, require_audio: bool = False) -> None:
    """Validate referential, temporal, formant, split, and representation invariants."""
    _validate_items(dataset.items, require_audio=require_audio)
    item_ids = set(dataset.items["item_id"].astype(str))
    durations = dataset.items.set_index("item_id")["duration_s"].astype(float).to_dict()

    _validate_tracks(dataset.tracks, item_ids, durations)
    _validate_intervals(dataset.intervals, item_ids, durations)
    interval_ids = set(dataset.intervals["interval_id"].astype(str))
    _validate_splits(dataset.splits, dataset.items, item_ids)
    _validate_static(dataset.static_measurements, item_ids, interval_ids, durations)
    _validate_gold_representation(dataset)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise DatasetValidationError(f"{table} is missing required column(s): {', '.join(missing)}")


def _validate_items(items: pd.DataFrame, *, require_audio: bool) -> None:
    _require_columns(items, ITEM_REQUIRED_COLUMNS, "items.parquet")
    if items.empty:
        raise DatasetValidationError("items.parquet must contain at least one item.")
    if items["item_id"].isna().any() or (items["item_id"].astype(str).str.len() == 0).any():
        raise DatasetValidationError("item_id values must be non-empty.")
    if items["item_id"].duplicated().any():
        raise DatasetValidationError("item_id values must be unique.")
    for field in ("source", "item_type"):
        if items[field].isna().any() or (items[field].astype(str).str.len() == 0).any():
            raise DatasetValidationError(f"items.parquet.{field} values must be non-empty.")
    durations = _numeric(items["duration_s"], "items.parquet.duration_s")
    if (durations <= 0).any():
        raise DatasetValidationError("Item duration_s must be greater than zero.")

    if require_audio:
        if "audio_path" not in items.columns:
            raise DatasetValidationError("items.parquet must include audio_path when audio resolution is required.")
        missing = [str(value) for value in items["audio_path"].dropna() if not Path(str(value)).is_file()]
        if items["audio_path"].isna().any() or missing:
            raise DatasetValidationError("Every item must reference an existing audio file when require_audio=True.")


def _validate_tracks(tracks: pd.DataFrame, item_ids: set[str], durations: dict[object, float]) -> None:
    _require_columns(tracks, TRACK_COLUMNS, "tracks.parquet")
    if tracks.empty:
        return
    _validate_references(tracks["item_id"], item_ids, "tracks.parquet.item_id")
    times = _numeric(tracks["time_s"], "tracks.parquet.time_s")
    if (times < 0).any():
        raise DatasetValidationError("Track time_s must be non-negative.")
    if tracks.duplicated(subset=["item_id", "time_s"]).any():
        raise DatasetValidationError("tracks.parquet contains duplicate (item_id, time_s) rows.")

    for item_id, group in tracks.assign(time_s=times).groupby("item_id", sort=False):
        item_times = group["time_s"]
        if not item_times.is_monotonic_increasing:
            raise DatasetValidationError(f"Track timestamps must be monotonic for item '{item_id}'.")
        duration = float(durations[item_id])
        if (item_times > duration + _TIME_TOLERANCE_S).any():
            raise DatasetValidationError(f"Track timestamps exceed duration for item '{item_id}'.")
    _validate_formant_columns(tracks, "tracks.parquet")


def _validate_intervals(intervals: pd.DataFrame, item_ids: set[str], durations: dict[object, float]) -> None:
    _require_columns(intervals, INTERVAL_COLUMNS, "intervals.parquet")
    if intervals.empty:
        return
    if intervals["interval_id"].isna().any() or intervals["interval_id"].duplicated().any():
        raise DatasetValidationError("interval_id values must be non-null and unique.")
    _validate_references(intervals["item_id"], item_ids, "intervals.parquet.item_id")
    starts = _numeric(intervals["start_s"], "intervals.parquet.start_s")
    ends = _numeric(intervals["end_s"], "intervals.parquet.end_s")
    if (starts < 0).any() or (ends <= starts).any():
        raise DatasetValidationError("Intervals require start_s >= 0 and end_s > start_s.")
    for row_index, (item_id, end_s) in enumerate(zip(intervals["item_id"], ends, strict=True)):
        if end_s > float(durations[item_id]) + _TIME_TOLERANCE_S:
            raise DatasetValidationError(f"Interval row {row_index} exceeds item duration.")
    valid_origins = {origin.value for origin in IntervalOrigin}
    if not intervals["origin"].isin(valid_origins).all():
        raise DatasetValidationError(f"Interval origin must be one of: {sorted(valid_origins)}")
    if intervals["interval_type"].isna().any() or (intervals["interval_type"].astype(str).str.len() == 0).any():
        raise DatasetValidationError("interval_type must be a non-empty string.")


def _validate_splits(splits: pd.DataFrame, items: pd.DataFrame, item_ids: set[str]) -> None:
    _require_columns(splits, SPLIT_COLUMNS, "splits.parquet")
    if splits.empty:
        return
    _validate_references(splits["item_id"], item_ids, "splits.parquet.item_id")
    if splits["split"].isna().any() or (splits["split"].astype(str).str.len() == 0).any():
        raise DatasetValidationError("Split labels must be non-empty.")
    if splits["item_id"].duplicated().any():
        raise DatasetValidationError("An item may belong to at most one prepared split.")
    validate_speaker_disjoint_splits(items, splits)


def _validate_static(
    static: pd.DataFrame | None,
    item_ids: set[str],
    interval_ids: set[str],
    durations: dict[object, float],
) -> None:
    if static is None:
        return
    _require_columns(static, STATIC_MEASUREMENT_COLUMNS, "static_measurements.parquet")
    if static.empty:
        return
    if static["measurement_id"].isna().any() or static["measurement_id"].duplicated().any():
        raise DatasetValidationError("measurement_id values must be non-null and unique.")
    if static["measurement_kind"].isna().any() or (static["measurement_kind"].astype(str).str.len() == 0).any():
        raise DatasetValidationError("measurement_kind values must be non-empty.")
    _validate_references(static["item_id"], item_ids, "static_measurements.parquet.item_id")

    referenced_intervals = static["interval_id"].dropna().astype(str)
    unknown_intervals = sorted(set(referenced_intervals) - interval_ids)
    if unknown_intervals:
        raise DatasetValidationError(f"Static measurements reference unknown interval_id(s): {unknown_intervals}")

    relative = _numeric_nullable(static["relative_position"], "static_measurements.parquet.relative_position")
    if ((relative < 0) | (relative > 1)).fillna(False).any():
        raise DatasetValidationError("relative_position must be between 0 and 1 when provided.")

    absolute = _numeric_nullable(static["time_s"], "static_measurements.parquet.time_s")
    if (absolute < 0).fillna(False).any():
        raise DatasetValidationError("Static time_s must be non-negative when provided.")

    window_start = _numeric_nullable(static["window_start_s"], "static_measurements.parquet.window_start_s")
    window_end = _numeric_nullable(static["window_end_s"], "static_measurements.parquet.window_end_s")
    partial_window = window_start.isna() ^ window_end.isna()
    if partial_window.any():
        raise DatasetValidationError("Static windows require both window_start_s and window_end_s.")
    invalid_window = ((window_start < 0) | (window_end <= window_start)).fillna(False)
    if invalid_window.any():
        raise DatasetValidationError("Static windows require start >= 0 and end > start.")

    for row_index, item_id in enumerate(static["item_id"]):
        duration = float(durations[item_id])
        if pd.notna(absolute.iloc[row_index]) and absolute.iloc[row_index] > duration + _TIME_TOLERANCE_S:
            raise DatasetValidationError(f"Static measurement row {row_index} time_s exceeds item duration.")
        if pd.notna(window_end.iloc[row_index]) and window_end.iloc[row_index] > duration + _TIME_TOLERANCE_S:
            raise DatasetValidationError(f"Static measurement row {row_index} window exceeds item duration.")
    _validate_formant_columns(static, "static_measurements.parquet")


def _validate_gold_representation(dataset: PreparedDataset) -> None:
    declared = {formant.value if isinstance(formant, Formant) else str(formant) for formant in dataset.manifest.available_formants}
    if not declared:
        raise DatasetValidationError("At least one available formant must be declared.")

    sources: list[pd.DataFrame] = []
    if dataset.manifest.annotation_type in {AnnotationType.TRACK, AnnotationType.MIXED}:
        if dataset.tracks.empty:
            raise DatasetValidationError("Track annotation_type requires non-empty tracks.parquet.")
        sources.append(dataset.tracks)
    if dataset.manifest.annotation_type in {AnnotationType.STATIC, AnnotationType.MIXED}:
        if dataset.static_measurements is None or dataset.static_measurements.empty:
            raise DatasetValidationError("Static annotation_type requires non-empty static_measurements.parquet.")
        sources.append(dataset.static_measurements)

    for formant in declared:
        if formant not in FORMANT_COLUMNS:
            raise DatasetValidationError(f"Unknown canonical formant '{formant}'.")
        if not any(formant in table.columns and table[formant].notna().any() for table in sources):
            raise DatasetValidationError(f"Declared available formant '{formant}' has no usable gold values.")

    undeclared_with_values = {
        formant
        for formant in FORMANT_COLUMNS
        if formant not in declared and any(formant in table.columns and table[formant].notna().any() for table in sources)
    }
    if undeclared_with_values:
        raise DatasetValidationError(
            "Gold values exist for undeclared formant(s): " + ", ".join(sorted(undeclared_with_values))
        )


def _validate_formant_columns(df: pd.DataFrame, table: str) -> None:
    for formant in FORMANT_COLUMNS:
        _numeric_nullable(df[formant], f"{table}.{formant}")


def _validate_references(values: pd.Series, valid: set[str], field: str) -> None:
    unknown = sorted(set(values.dropna().astype(str)) - valid)
    if values.isna().any() or unknown:
        detail = f" Unknown values: {unknown}" if unknown else ""
        raise DatasetValidationError(f"{field} contains missing or unknown references.{detail}")


def _numeric(series: pd.Series, field: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise DatasetValidationError(f"{field} must contain finite numeric values.")
    return converted.astype(float)


def _numeric_nullable(series: pd.Series, field: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & converted.isna()
    finite = np.isfinite(converted.fillna(0).to_numpy(dtype=float))
    if invalid.any() or not finite.all():
        raise DatasetValidationError(f"{field} must contain finite numeric or missing values.")
    return converted.astype(float)
