"""Root-mean-squared formant error."""

from __future__ import annotations

import numpy as np

from formant_benchmark.evaluation.metrics._common import matched_values


def rmse(gold: np.ndarray, prediction: np.ndarray) -> float:
    """Compute RMSE over finite matched values, or NaN when none match."""
    gold_values, prediction_values = matched_values(gold, prediction)
    if not len(gold_values):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(gold_values - prediction_values))))
