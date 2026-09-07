"""Prediction extraction for source-defined static formant observations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from formant_benchmark.evaluation.align import interpolate_formant, window_time_mean
from formant_benchmark.exceptions import EvaluationCompatibilityError


def static_prediction(
    measurement: pd.Series | dict[str, Any],
    prediction: pd.DataFrame,
    *,
    formant: str,
    interval: pd.Series | dict[str, Any] | None,
    item_duration_s: float,
) -> float:
    """Derive one tracker value using the source-defined static location semantics."""
    row = dict(measurement)
    window_start = _optional_float(row.get("window_start_s"))
    window_end = _optional_float(row.get("window_end_s"))
    if window_start is not None or window_end is not None:
        if window_start is None or window_end is None:
            raise EvaluationCompatibilityError("Static measurement window bounds must be supplied together.")
        return window_time_mean(
            prediction,
            start_s=window_start,
            end_s=window_end,
            formant=formant,
        )

    time_s = _optional_float(row.get("time_s"))
    if time_s is None:
        relative = _optional_float(row.get("relative_position"))
        if relative is not None:
            start_s, end_s = _reference_bounds(interval, item_duration_s)
            time_s = start_s + relative * (end_s - start_s)
        else:
            start_s, end_s = _reference_bounds(interval, item_duration_s)
            time_s = (start_s + end_s) / 2.0

    return float(interpolate_formant(prediction, np.asarray([time_s]), formant)[0])


def _reference_bounds(
    interval: pd.Series | dict[str, Any] | None,
    item_duration_s: float,
) -> tuple[float, float]:
    if interval is None:
        return 0.0, float(item_duration_s)
    row = dict(interval)
    return float(row["start_s"]), float(row["end_s"])


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
