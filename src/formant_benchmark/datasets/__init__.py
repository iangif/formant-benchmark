"""Dataset adapter extension points and built-in source adapters."""

from formant_benchmark.datasets.base import DATASET_REGISTRY, DatasetAdapter
from formant_benchmark.datasets.mcqll_formants import MCQLLFormantsAdapter


def register_builtin_datasets() -> None:
    """Register benchmark-provided adapters without replacing existing entries."""
    if MCQLLFormantsAdapter.name not in DATASET_REGISTRY.names():
        DATASET_REGISTRY.register(MCQLLFormantsAdapter.name, MCQLLFormantsAdapter)


__all__ = [
    "DATASET_REGISTRY",
    "DatasetAdapter",
    "MCQLLFormantsAdapter",
    "register_builtin_datasets",
]
