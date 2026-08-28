"""Source-specific dataset adapter contract and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from formant_benchmark.data.models import PreparedDataset
from formant_benchmark.registry import Registry


class DatasetAdapter(ABC):
    """Convert one source format into the normalized PreparedDataset contract."""

    name: str

    @abstractmethod
    def prepare(self, config: Mapping[str, Any]) -> PreparedDataset:
        """Parse source-specific data and return normalized in-memory tables."""
        raise NotImplementedError


DATASET_REGISTRY: Registry[type[DatasetAdapter]] = Registry()
