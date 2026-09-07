"""Prediction coverage over eligible gold measurements."""

from __future__ import annotations

import numpy as np

from formant_benchmark.evaluation.metrics._common import eligible_gold_mask


def coverage(gold: np.ndarray, prediction: np.ndarray) -> float:
    """Return the fraction of finite gold observations with finite predictions."""
    gold_values = np.asarray(gold, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    eligible = eligible_gold_mask(gold_values)
    denominator = int(eligible.sum())
    if denominator == 0:
        return float("nan")
    matched = eligible & np.isfinite(prediction_values)
    return float(matched.sum() / denominator)
