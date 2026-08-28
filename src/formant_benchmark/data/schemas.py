"""Canonical prepared-dataset table schemas and empty-table constructors."""

from __future__ import annotations

import pandas as pd

FORMANT_COLUMNS = ("F1", "F2", "F3", "F4")

ITEM_COLUMNS = (
    "item_id",
    "source",
    "item_type",
    "speaker_id",
    "gender",
    "age",
    "language",
    "dialect",
    "audio_path",
    "duration_s",
)
ITEM_REQUIRED_COLUMNS = ("item_id", "source", "item_type", "duration_s")

TRACK_COLUMNS = ("item_id", "time_s", *FORMANT_COLUMNS)

INTERVAL_COLUMNS = (
    "interval_id",
    "item_id",
    "interval_type",
    "label",
    "start_s",
    "end_s",
    "origin",
)

SPLIT_COLUMNS = ("item_id", "split")

STATIC_MEASUREMENT_COLUMNS = (
    "measurement_id",
    "item_id",
    "interval_id",
    "measurement_kind",
    "relative_position",
    "time_s",
    "window_start_s",
    "window_end_s",
    *FORMANT_COLUMNS,
)


def empty_items() -> pd.DataFrame:
    """Return an empty items table with canonical common columns."""
    return pd.DataFrame(columns=ITEM_COLUMNS)


def empty_tracks() -> pd.DataFrame:
    """Return an empty track table with canonical F1-F4 columns."""
    return pd.DataFrame(columns=TRACK_COLUMNS)


def empty_intervals() -> pd.DataFrame:
    """Return an empty interval table."""
    return pd.DataFrame(columns=INTERVAL_COLUMNS)


def empty_splits() -> pd.DataFrame:
    """Return an empty split table."""
    return pd.DataFrame(columns=SPLIT_COLUMNS)


def empty_static_measurements() -> pd.DataFrame:
    """Return an empty static-measurement table with canonical F1-F4 columns."""
    return pd.DataFrame(columns=STATIC_MEASUREMENT_COLUMNS)
