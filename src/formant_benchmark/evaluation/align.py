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

    # Vectorised over targets; each target follows the same rules independently.
    insertion = np.searchsorted(times, targets, side="left")
    right = np.minimum(insertion, len(times) - 1)
    left = np.maximum(insertion - 1, 0)
    exact_right = (insertion < len(times)) & np.isclose(times[right], targets, atol=_TIME_ATOL, rtol=0)
    exact_left = ~exact_right & (insertion > 0) & np.isclose(times[left], targets, atol=_TIME_ATOL, rtol=0)
    result[exact_right] = values[right][exact_right]
    result[exact_left] = values[left][exact_left]

    between = (
        ~exact_right
        & ~exact_left
        & (insertion > 0)
        & (insertion < len(times))
        & np.isfinite(values[left])
        & np.isfinite(values[right])
        & (targets >= times[left])
        & (targets <= times[right])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = (targets - times[left]) / (times[right] - times[left])
        interpolated = values[left] + fraction * (values[right] - values[left])
    result[between] = interpolated[between]
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
