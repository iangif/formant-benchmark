"""Prediction-to-gold temporal alignment with no extrapolation or NaN bridging."""

from __future__ import annotations

import numpy as np
import pandas as pd


_TIME_ATOL = 1e-9


def interpolate_formant(
    prediction: pd.DataFrame,
    target_times: np.ndarray,
    formant: str,
) -> np.ndarray:
    """Linearly interpolate one formant onto target times without extrapolation.

    Missing prediction samples are treated as genuine gaps: interpolation is only
    performed between adjacent finite samples. An exact target timestamp inherits
    the stored value, including NaN.
    """
    targets = np.asarray(target_times, dtype=float)
    result = np.full(targets.shape, np.nan, dtype=float)
    if prediction.empty:
        return result

    ordered = prediction.sort_values("time_s", kind="stable")
    times = pd.to_numeric(ordered["time_s"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(ordered[formant], errors="coerce").to_numpy(dtype=float)
    if not len(times):
        return result

    for index, target in np.ndenumerate(targets):
        insertion = int(np.searchsorted(times, target, side="left"))
        exact_index: int | None = None
        if insertion < len(times) and np.isclose(times[insertion], target, atol=_TIME_ATOL, rtol=0):
            exact_index = insertion
        elif insertion > 0 and np.isclose(times[insertion - 1], target, atol=_TIME_ATOL, rtol=0):
            exact_index = insertion - 1
        if exact_index is not None:
            if np.isfinite(values[exact_index]):
                result[index] = values[exact_index]
            continue

        if insertion == 0 or insertion >= len(times):
            continue
        left = insertion - 1
        right = insertion
        if not (np.isfinite(values[left]) and np.isfinite(values[right])):
            continue
        if target < times[left] or target > times[right]:
            continue
        fraction = (target - times[left]) / (times[right] - times[left])
        result[index] = values[left] + fraction * (values[right] - values[left])
    return result


def window_time_mean(
    prediction: pd.DataFrame,
    *,
    start_s: float,
    end_s: float,
    formant: str,
) -> float:
    """Return the piecewise-linear time mean over a fully supported window.

    The prediction must provide finite support at both window boundaries and may
    not contain a missing stored sample inside the window. This avoids silently
    bridging a tracker gap while making the result insensitive to frame spacing.
    """
    if end_s <= start_s or prediction.empty:
        return float("nan")
    ordered = prediction.sort_values("time_s", kind="stable")
    times = pd.to_numeric(ordered["time_s"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(ordered[formant], errors="coerce").to_numpy(dtype=float)
    interior = (times > start_s + _TIME_ATOL) & (times < end_s - _TIME_ATOL)
    if np.isnan(values[interior]).any():
        return float("nan")

    boundary_values = interpolate_formant(
        ordered,
        np.asarray([start_s, end_s], dtype=float),
        formant,
    )
    if not np.isfinite(boundary_values).all():
        return float("nan")

    inside_times = times[interior]
    inside_values = values[interior]
    if not np.isfinite(inside_values).all():
        return float("nan")
    sample_times = np.concatenate(([start_s], inside_times, [end_s]))
    sample_values = np.concatenate(([boundary_values[0]], inside_values, [boundary_values[1]]))
    integral = np.sum(np.diff(sample_times) * (sample_values[:-1] + sample_values[1:]) / 2.0)
    return float(integral / (end_s - start_s))
