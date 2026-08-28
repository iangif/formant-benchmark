"""Milestone tests: construct, save, reload, inspect, and validate synthetic datasets."""

from pathlib import Path
import importlib.util

import pytest

from formant_benchmark.data.io import inspect_prepared_dataset, load_prepared_dataset, write_prepared_dataset
from formant_benchmark.exceptions import DatasetAlreadyExistsError
from tests.fixtures.synthetic import static_dataset, trajectory_dataset

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None,
    reason="Parquet engine is not installed in the test environment.",
)


def test_trajectory_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "prepared" / "trajectory"
    persisted = write_prepared_dataset(trajectory_dataset(), destination)
    loaded = load_prepared_dataset(destination)

    assert persisted.manifest.fingerprint
    assert loaded.manifest.fingerprint == persisted.manifest.fingerprint
    assert (destination / "dataset.yaml").is_file()
    assert (destination / "items.parquet").is_file()
    assert (destination / "tracks.parquet").is_file()
    assert (destination / "intervals.parquet").is_file()
    assert (destination / "splits.parquet").is_file()
    assert not (destination / "static_measurements.parquet").exists()

    summary = inspect_prepared_dataset(loaded)
    assert summary["n_items"] == 2
    assert summary["available_formants"] == ["F1", "F2", "F3", "F4"]


def test_static_round_trip_and_safe_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "prepared" / "static"
    write_prepared_dataset(static_dataset(), destination)
    assert (destination / "static_measurements.parquet").is_file()

    with pytest.raises(DatasetAlreadyExistsError):
        write_prepared_dataset(static_dataset(), destination)

    replacement = static_dataset()
    replacement.items.loc[0, "age"] = 23
    write_prepared_dataset(replacement, destination, overwrite=True)
    loaded = load_prepared_dataset(destination)
    assert loaded.items.loc[0, "age"] == 23
    assert inspect_prepared_dataset(loaded)["n_static_measurements"] == 3
