"""Parquet/YAML persistence for prepared datasets with safe replacement semantics."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from formant_benchmark.data.models import AnnotationType, DatasetManifest, PreparedDataset
from formant_benchmark.data.schemas import empty_static_measurements
from formant_benchmark.exceptions import (
    DatasetAlreadyExistsError,
    DatasetFingerprintMismatchError,
    DatasetValidationError,
)
from formant_benchmark.preparation.fingerprint import dataset_fingerprint
from formant_benchmark.preparation.validation import validate_prepared_dataset


def write_prepared_dataset(
    dataset: PreparedDataset,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> PreparedDataset:
    """Validate and atomically persist a prepared dataset directory.

    Existing destinations fail by default. With ``overwrite=True``, the replacement
    is fully written, reloaded, and validated before the previous directory is moved.
    """
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        raise DatasetAlreadyExistsError(f"Prepared dataset destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    validate_prepared_dataset(dataset)
    fingerprint = dataset_fingerprint(dataset)
    manifest = dataset.manifest.model_copy(update={"fingerprint": fingerprint})
    persisted = PreparedDataset(
        manifest=manifest,
        items=dataset.items.copy(),
        tracks=dataset.tracks.copy(),
        intervals=dataset.intervals.copy(),
        splits=dataset.splits.copy(),
        static_measurements=dataset.static_measurements.copy() if dataset.static_measurements is not None else None,
        root=destination_path,
    )

    temp_path = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.tmp-", dir=destination_path.parent))
    backup_path: Path | None = None
    try:
        _write_directory(persisted, temp_path)
        load_prepared_dataset(temp_path, validate=True)

        if destination_path.exists():
            backup_path = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.backup-", dir=destination_path.parent))
            backup_path.rmdir()
            os.replace(destination_path, backup_path)
        os.replace(temp_path, destination_path)
        if backup_path is not None:
            shutil.rmtree(backup_path, ignore_errors=True)
    except Exception:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        if backup_path is not None and backup_path.exists() and not destination_path.exists():
            os.replace(backup_path, destination_path)
        raise
    return persisted


def load_prepared_dataset(path: str | Path, *, validate: bool = True) -> PreparedDataset:
    """Load a prepared dataset and verify its recorded fingerprint when present."""
    root = Path(path)
    try:
        manifest_raw = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
        manifest = DatasetManifest.model_validate(manifest_raw)
        dataset = PreparedDataset(
            manifest=manifest,
            items=pd.read_parquet(root / "items.parquet"),
            tracks=pd.read_parquet(root / "tracks.parquet"),
            intervals=pd.read_parquet(root / "intervals.parquet"),
            splits=pd.read_parquet(root / "splits.parquet"),
            static_measurements=_read_static_if_present(root, manifest),
            root=root,
        )
    except (OSError, ValueError, TypeError, ImportError) as exc:
        raise DatasetValidationError(f"Could not load prepared dataset '{root}': {exc}") from exc

    if validate:
        validate_prepared_dataset(dataset)
        if manifest.fingerprint:
            actual = dataset_fingerprint(dataset)
            if actual != manifest.fingerprint:
                raise DatasetFingerprintMismatchError(
                    f"Prepared dataset fingerprint mismatch: recorded={manifest.fingerprint}, actual={actual}"
                )
    return dataset


def inspect_prepared_dataset(dataset: PreparedDataset) -> dict[str, Any]:
    """Return a compact programmatic summary suitable for later CLI inspection."""
    interval_counts = (
        dataset.intervals["interval_type"].value_counts(dropna=False).sort_index().to_dict()
        if not dataset.intervals.empty
        else {}
    )
    split_counts = dataset.splits["split"].value_counts(dropna=False).sort_index().to_dict() if not dataset.splits.empty else {}
    return {
        "name": dataset.manifest.name,
        "source": dataset.manifest.source,
        "annotation_type": dataset.manifest.annotation_type.value,
        "available_formants": [formant.value for formant in dataset.manifest.available_formants],
        "fingerprint": dataset.manifest.fingerprint or dataset_fingerprint(dataset),
        "n_items": int(len(dataset.items)),
        "n_track_rows": int(len(dataset.tracks)),
        "n_intervals": int(len(dataset.intervals)),
        "n_static_measurements": int(len(dataset.static_measurements)) if dataset.static_measurements is not None else 0,
        "interval_counts": {str(key): int(value) for key, value in interval_counts.items()},
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
    }


def _write_directory(dataset: PreparedDataset, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_data = dataset.manifest.model_dump(mode="json")
    (root / "dataset.yaml").write_text(yaml.safe_dump(manifest_data, sort_keys=False), encoding="utf-8")
    dataset.items.to_parquet(root / "items.parquet", index=False)
    dataset.tracks.to_parquet(root / "tracks.parquet", index=False)
    dataset.intervals.to_parquet(root / "intervals.parquet", index=False)
    dataset.splits.to_parquet(root / "splits.parquet", index=False)
    if dataset.manifest.annotation_type in {AnnotationType.STATIC, AnnotationType.MIXED}:
        static = dataset.static_measurements if dataset.static_measurements is not None else empty_static_measurements()
        static.to_parquet(root / "static_measurements.parquet", index=False)


def _read_static_if_present(root: Path, manifest: DatasetManifest) -> pd.DataFrame | None:
    static_path = root / "static_measurements.parquet"
    if static_path.exists():
        return pd.read_parquet(static_path)
    if manifest.annotation_type in {AnnotationType.STATIC, AnnotationType.MIXED}:
        raise DatasetValidationError("Prepared static/mixed dataset is missing static_measurements.parquet.")
    return None
