"""Minimal tracker capability types and registry used by later integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.registry import Registry


@dataclass(froze=True, slots=True)
class TrackerCapabilities:
    """Static capabilities exposed before expensive tracker execution."""

    formants: frozenset[Formant]
    input_modes: frozenset[TrackingInputMode]
    interval_types: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.formants:
            raise ValueError("A tracker must support at least one canonical formant.")


class TrackerAdapter(ABC):
    """Batch-oriented public tracker contract; concrete execution is deferred."""

    name: str
    capabilities: TrackerCapabilities

    @abstractmethod
    def run(self, inputs: Iterable[Mapping[str, Any]]) -> Any:
        """Run a collection of normalized tracking inputs."""
        raise NotImplementedError


TRACKER_REGISTRY: Registry[type[TrackerAdapter]] = Registry()
