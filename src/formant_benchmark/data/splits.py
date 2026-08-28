"""Validation helpers for experimental dataset splits."""

from __future__ import annotations

import pandas as pd

from formant_benchmark.exceptions import DatasetValidationError

TRAIN_DEV_TEST = frozenset({"train", "dev", "test"})


def validate_speaker_disjoint_splits(items: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Reject speaker leakage across train/dev/test while allowing other split labels."""
    if splits.empty:
        return
    if "speaker_id" not in items.columns:
        return

    merged = splits.merge(items[["item_id", "speaker_id"]], on="item_id", how="left", validate="many_to_one")
    core = merged[merged["split"].isin(TRAIN_DEV_TEST)].dropna(subset=["speaker_id"])
    if core.empty:
        return

    per_speaker = core.groupby("speaker_id", dropna=False)["split"].nunique()
    leaking = sorted(str(speaker) for speaker in per_speaker[per_speaker > 1].index)
    if leaking:
        raise DatasetValidationError(
            "Train/dev/test splits must be speaker-disjoint; speaker(s) appear in multiple splits: "
            + ", ".join(leaking)
        )
