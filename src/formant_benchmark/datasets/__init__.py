"""Dataset adapter extension points and built-in source adapters."""

from formant_benchmark.datasets.base import DATASET_REGISTRY, DatasetAdapter
from formant_benchmark.datasets.hillenbrand import HillenbrandAdapter
from formant_benchmark.datasets.mcqll_formants import MCQLLFormantsAdapter
from formant_benchmark.datasets.vtr import VTRAdapter


def register_builtin_datasets() -> None:
    """Register benchmark-provided adapters without replacing existing entries."""
    for adapter in (HillenbrandAdapter, MCQLLFormantsAdapter, VTRAdapter):
        if adapter.name not in DATASET_REGISTRY.names():
            DATASET_REGISTRY.register(adapter.name, adapter)


__all__ = [
    "DATASET_REGISTRY",
    "DatasetAdapter",
    "HillenbrandAdapter",
    "MCQLLFormantsAdapter",
    "VTRAdapter",
    "register_builtin_datasets",
]
