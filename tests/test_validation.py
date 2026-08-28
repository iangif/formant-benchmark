"""Tests for structural prepared-dataset validation."""

import pytest

from formant_benchmark.exceptions import DatasetValidationError
from formant_benchmark.preparation.validation import validate_prepared_dataset
from tests.fixtures.synthetic import static_dataset, trajectory_dataset


def test_synthetic_trajectory_and_static_datasets_validate() -> None:
    validate_prepared_dataset(trajectory_dataset())
    validate_prepared_dataset(static_dataset())


def test_duplicate_items_fail() -> None:
    dataset = trajectory_dataset()
    dataset.items.loc[1, "item_id"] = dataset.items.loc[0, "item_id"]
    with pytest.raises(DatasetValidationError, match="item_id values must be unique"):
        validate_prepared_dataset(dataset)


def test_track_outside_item_duration_fails() -> None:
    dataset = trajectory_dataset()
    dataset.tracks.loc[0, "time_s"] = 1.0
    with pytest.raises(DatasetValidationError, match="monotonic|exceed"):
        validate_prepared_dataset(dataset)


def test_speaker_leakage_across_train_test_fails() -> None:
    dataset = trajectory_dataset()
    dataset.items.loc[1, "speaker_id"] = "spk-a"
    with pytest.raises(DatasetValidationError, match="speaker-disjoint"):
        validate_prepared_dataset(dataset)


def test_partial_static_window_fails() -> None:
    dataset = static_dataset()
    dataset.static_measurements.loc[0, "window_start_s"] = 0.1
    with pytest.raises(DatasetValidationError, match="both window_start_s and window_end_s"):
        validate_prepared_dataset(dataset)


def test_single_formant_gold_is_valid() -> None:
    from formant_benchmark.data.models import Formant

    dataset = trajectory_dataset()
    dataset.manifest.available_formants = [Formant.F2]
    for formant in ("F1", "F3", "F4"):
        dataset.tracks[formant] = None
    validate_prepared_dataset(dataset)


def test_structural_validation_does_not_apply_acoustic_cleaning() -> None:
    dataset = trajectory_dataset()
    dataset.tracks.loc[0, "F1"] = 2200.0
    dataset.tracks.loc[0, "F2"] = 1200.0
    validate_prepared_dataset(dataset)


def test_source_provided_voiced_intervals_can_be_preserved() -> None:
    dataset = trajectory_dataset()
    voiced = dataset.intervals.iloc[[0]].copy()
    voiced["interval_id"] = "utt-1:voiced-1"
    voiced["interval_type"] = "voiced"
    voiced["origin"] = "source"
    dataset.intervals = __import__("pandas").concat([dataset.intervals, voiced], ignore_index=True)
    validate_prepared_dataset(dataset)
