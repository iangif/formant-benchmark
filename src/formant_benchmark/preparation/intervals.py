"""Generic interval transforms driven by dataset-adapter knowledge."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from formant_benchmark.data.models import IntervalOrigin
from formant_benchmark.data.schemas import INTERVAL_COLUMNS
from formant_benchmark.exceptions import DatasetValidationError


def derive_vowel_intervals(intervals: pd.DataFrame, vowel_labels: Collection[str]) -> pd.DataFrame:
    """Derive standardized vowel intervals from phone intervals and adapter labels.

    Existing intervals are preserved. Each derived vowel receives a stable ID based
    on its source phone interval, making the transform deterministic and idempotent.
    """
    missing = [column for column in INTERVAL_COLUMNS if column not in intervals.columns]
    if missing:
        raise DatasetValidationError(f"Cannot derive vowels; interval columns missing: {missing}")

    labels = set(vowel_labels)
    if not labels:
        return intervals.copy()

    phones = intervals[(intervals["interval_type"] == "phone") & intervals["label"].isin(labels)].copy()
    if phones.empty:
        return intervals.copy()

    phones["interval_id"] = phones["interval_id"].astype(str).map(lambda value: f"{value}:vowel")
    phones["interval_type"] = "vowel"
    phones["origin"] = IntervalOrigin.DERIVED.value

    existing_ids = set(intervals["interval_id"].astype(str))
    phones = phones[~phones["interval_id"].isin(existing_ids)]
    return pd.concat([intervals, phones], ignore_index=True)
