"""Tests for deterministic, machine-path-independent prepared-dataset identity."""

from formant_benchmark.preparation.fingerprint import dataset_fingerprint
from tests.fixtures.synthetic import trajectory_dataset


def test_fingerprint_ignores_row_order_and_absolute_source_root() -> None:
    first = trajectory_dataset(source_root="/machine-a/corpora/synthetic")
    second = trajectory_dataset(source_root="/different-machine/data/synthetic")
    second.items = second.items.iloc[::-1].reset_index(drop=True)
    second.tracks = second.tracks.iloc[::-1].reset_index(drop=True)
    assert dataset_fingerprint(first) == dataset_fingerprint(second)


def test_fingerprint_changes_when_normalized_gold_changes() -> None:
    first = trajectory_dataset()
    second = trajectory_dataset()
    second.tracks.loc[0, "F1"] += 1
    assert dataset_fingerprint(first) != dataset_fingerprint(second)


def test_fingerprint_ignores_set_like_batch_order() -> None:
    first = trajectory_dataset()
    second = trajectory_dataset()
    first.manifest.preparation_config["batches"] = ["batch1", "batch2"]
    second.manifest.preparation_config["batches"] = ["batch2", "batch1"]
    assert dataset_fingerprint(first) == dataset_fingerprint(second)


def test_fingerprint_ignores_tracker_tuning() -> None:
    first = trajectory_dataset()
    second = trajectory_dataset()
    second.manifest.tracker_overrides = {"synthetic": {"parameters": {"offset_hz": 100}}}
    assert dataset_fingerprint(first) == dataset_fingerprint(second)
