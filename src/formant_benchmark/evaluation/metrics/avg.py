"""Signed average formant difference (gold minus prediction)."""

from __future__ import annotations

import numpy as np

from formant_benchmark.evaluation.metrics._common import matched_values


def avg(gold: np.ndarray, prediction: np.ndarray) -> float:
    """Compute mean(gold - prediction), or NaN when no values match."""
    gold_values, prediction_values = matched_values(gold, prediction)
    if not len(gold_values):
        return float("nan")
    return float(np.mean(gold_values - prediction_values))
