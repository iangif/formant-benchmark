"""Tracker-side wrapper protocol and dependency-isolated entry points.

The package initializer stays dependency-free so isolated wrappers can start
without importing the benchmark's dataframe stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "normalize_wrapper_rows",
    "read_wrapper_output",
    "write_wrapper_output",
    "write_wrapper_request",
]


def __getattr__(name: str) -> Any:
    """Lazily retain the public protocol helpers from the original API."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from formant_benchmark.tracker_wrappers import protocol

    value = getattr(protocol, name)
    globals()[name] = value
    return value
