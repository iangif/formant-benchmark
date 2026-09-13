"""Tracker interfaces, built-in adapters, and registry initialization."""

from formant_benchmark.trackers.base import (
    TRACKER_REGISTRY,
    TrackerAdapter,
    TrackerCapabilities,
    TrackingInput,
)
from formant_benchmark.trackers.fasttrackpy import FastTrackPyTracker
from formant_benchmark.trackers.synthetic import SyntheticTracker


def register_builtin_trackers() -> None:
    """Register benchmark-provided trackers without replacing extensions."""
    for tracker_type in (FastTrackPyTracker, SyntheticTracker):
        if tracker_type.name not in TRACKER_REGISTRY.names():
            TRACKER_REGISTRY.register(tracker_type.name, tracker_type)


__all__ = [
    "TRACKER_REGISTRY",
    "FastTrackPyTracker",
    "SyntheticTracker",
    "TrackerAdapter",
    "TrackerCapabilities",
    "TrackingInput",
    "register_builtin_trackers",
]
