"""Exercise prepared-dataset I/O orchestration when a Parquet engine is unavailable."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from formant_benchmark.data.io import load_prepared_dataset, write_prepared_dataset
from tests.fixtures.synthetic import static_dataset, trajectory_dataset


def _install_pickle_parquet_shim(monkeypatch) -> None:
    """Substitute only the external serialization backend, not benchmark I/O logic."""
    def to_parquet(self, path, index=False, **_kwargs):
        frame = self.reset_index(drop=True) if not index else self
        frame.to_pickle(path)

    def read_parquet(path, **_kwargs):
        return pd.read_pickle(path)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet)
    monkeypatch.setattr(pd, "read_parquet", read_parquet)


def test_round_trip_orchestration_with_backend_shim(tmp_path: Path, monkeypatch) -> None:
    _install_pickle_parquet_shim(monkeypatch)
    destination = tmp_path / "trajectory"
    written = write_prepared_dataset(trajectory_dataset(), destination)
    loaded = load_prepared_dataset(destination)
    assert loaded.manifest.fingerprint == written.manifest.fingerprint
    assert len(loaded.tracks) == 10


def test_static_safe_overwrite_with_backend_shim(tmp_path: Path, monkeypatch) -> None:
    _install_pickle_parquet_shim(monkeypatch)
    destination = tmp_path / "static"
    write_prepared_dataset(static_dataset(), destination)
    replacement = static_dataset()
    replacement.items.loc[0, "age"] = 99
    write_prepared_dataset(replacement, destination, overwrite=True)
    assert load_prepared_dataset(destination).items.loc[0, "age"] == 99
