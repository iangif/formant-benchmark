"""Formant Detection Rate under absolute and relative error thresholds."""

from __future__ import annotations

import numpy as np

from formant_benchmark.evaluation.metrics._common import eligible_gold_mask


def fdr(
    gold: np.ndarray,
    prediction: np.ndarray,
    *,
    relative_threshold: float = 0.30,
    absolute_threshold_hz: float = 300.0,
) -> float:
    """Return threshold successes divided by all eligible gold observations.

    Missing predictions stay in the denominator and therefore count as misses.
    Both the relative and absolute criteria must be satisfied.
    """
    gold_values = np.asarray(gold, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    eligible = eligible_gold_mask(gold_values)
    denominator = int(eligible.sum())
    if denominator == 0:
        return float("nan")

    matched = eligible & np.isfinite(prediction_values)
    absolute_error = np.full(gold_values.shape, np.inf, dtype=float)
    relative_error = np.full(gold_values.shape, np.inf, dtype=float)
    absolute_error[matched] = np.abs(gold_values[matched] - prediction_values[matched])
    nonzero = matched & (np.abs(gold_values) > 0)
    relative_error[nonzero] = absolute_error[nonzero] / np.abs(gold_values[nonzero])
    exact_zero = matched & (gold_values == 0) & (prediction_values == 0)
    relative_error[exact_zero] = 0.0
    successes = matched & (absolute_error < absolute_threshold_hz) & (relative_error < relative_threshold)
    return float(successes.sum() / denominator)
