"""Tracker interfaces, built-in adapters, and registry initialization."""

from formant_benchmark.trackers.base import (
    TRACKER_REGISTRY,
    TrackerAdapter,
    TrackerCapabilities,
    TrackingInput,
)
from formant_benchmark.trackers.synthetic import SyntheticTracker


def register_builtin_trackers() -> None:
    """Register benchmark-provided trackers without replacing extensions."""
    if SyntheticTracker.name not in TRACKER_REGISTRY.names():
        TRACKER_REGISTRY.register(SyntheticTracker.name, SyntheticTracker)


__all__ = [
    "TRACKER_REGISTRY",
    "SyntheticTracker",
    "TrackerAdapter",
    "TrackerCapabilities",
    "TrackingInput",
    "register_builtin_trackers",
]
