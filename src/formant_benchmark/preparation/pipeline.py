"""Generic preparation orchestration shared by future source-specific adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from formant_benchmark.data.io import write_prepared_dataset
from formant_benchmark.data.models import PreparedDataset
from formant_benchmark.datasets.base import DatasetAdapter
from formant_benchmark.preparation.validation import validate_prepared_dataset


def prepare_dataset(
    adapter: DatasetAdapter,
    config: Mapping[str, Any],
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> PreparedDataset:
    """Run an adapter, validate normalized output, and persist it safely."""
    dataset = adapter.prepare(config)
    validate_prepared_dataset(dataset)
    return write_prepared_dataset(dataset, destination, overwrite=overwrite)
