"""Dependency-light JSON protocol shared across tracker environment boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from formant_benchmark.data.schemas import FORMANT_COLUMNS, PREDICTION_COLUMNS
from formant_benchmark.exceptions import TrackerExecutionError


def write_wrapper_request(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic wrapper request."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def write_wrapper_output(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write normalized rows as newline-delimited JSON."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=_json_default) + "\n")


def read_wrapper_output(
    path: Path,
    *,
    item_id: str,
    duration_s: float,
    supported_formants: set[str],
) -> pd.DataFrame:
    """Read and structurally validate normalized wrapper output."""
    if not path.is_file():
        raise TrackerExecutionError("Tracker wrapper did not create its requested output file.")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not a JSON object")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrackerExecutionError(f"Tracker wrapper output is unreadable: {exc}") from exc

    return normalize_wrapper_rows(
        rows,
        item_id=item_id,
        duration_s=duration_s,
        supported_formants=supported_formants,
    )


def normalize_wrapper_rows(
    rows: list[Mapping[str, Any]],
    *,
    item_id: str,
    duration_s: float,
    supported_formants: set[str],
) -> pd.DataFrame:
    """Validate in-memory wrapper rows and return the prediction schema."""
    if not rows:
        raise TrackerExecutionError("Tracker wrapper produced no prediction rows.")
    if any(not isinstance(row, Mapping) for row in rows):
        raise TrackerExecutionError("Tracker wrapper prediction rows must be JSON objects.")
    frame = pd.DataFrame([dict(row) for row in rows])
    if "time_s" not in frame.columns:
        raise TrackerExecutionError("Tracker wrapper output is missing required column 'time_s'.")
    if "item_id" in frame.columns:
        values = set(frame["item_id"].dropna().astype(str))
        if values and values != {item_id}:
            raise TrackerExecutionError("Tracker wrapper output contains an unexpected item_id.")

    times = pd.to_numeric(frame["time_s"], errors="coerce")
    if times.isna().any() or not np.isfinite(times.to_numpy(dtype=float)).all():
        raise TrackerExecutionError("Tracker wrapper time_s values must be finite numbers.")
    if (times < -1e-6).any() or (times > duration_s + 1e-6).any():
        raise TrackerExecutionError("Tracker wrapper timestamps lie outside the tracking input.")
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise TrackerExecutionError("Tracker wrapper timestamps must be strictly increasing.")

    normalized = pd.DataFrame({"item_id": item_id, "time_s": times.astype(float)})
    for formant in FORMANT_COLUMNS:
        if formant not in frame.columns:
            normalized[formant] = pd.Series([None] * len(frame), dtype="Float64")
            continue
        values = pd.to_numeric(frame[formant], errors="coerce")
        invalid = frame[formant].notna() & values.isna()
        if invalid.any() or not np.isfinite(values.fillna(0).to_numpy(dtype=float)).all():
            raise TrackerExecutionError(f"Tracker wrapper {formant} values must be numeric or missing.")
        if formant not in supported_formants and values.notna().any():
            raise TrackerExecutionError(f"Tracker emitted undeclared formant {formant}.")
        normalized[formant] = values.astype("Float64")

    if not any(normalized[formant].notna().any() for formant in supported_formants):
        raise TrackerExecutionError("Tracker wrapper output contains no usable declared formant values.")
    return normalized.loc[:, PREDICTION_COLUMNS]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")
