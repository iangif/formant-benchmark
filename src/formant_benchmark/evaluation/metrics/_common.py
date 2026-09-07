"""Shared finite-value selection for numerical evaluation metrics."""

from __future__ import annotations

import numpy as np


def matched_values(gold: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite gold/prediction pairs from already aligned arrays."""
    gold_values = np.asarray(gold, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    mask = np.isfinite(gold_values) & np.isfinite(prediction_values)
    return gold_values[mask], prediction_values[mask]


def eligible_gold_mask(gold: np.ndarray) -> np.ndarray:
    """Return positions that contain a usable gold observation."""
    return np.isfinite(np.asarray(gold, dtype=float))
