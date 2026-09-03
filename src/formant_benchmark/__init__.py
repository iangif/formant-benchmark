"""Core package for the formant measurement benchmarking suite.

Public model exports are loaded lazily so tracker-side wrapper processes do not
pay the pandas/Pydantic import cost merely by importing a lightweight wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from formant_benchmark.data.models import (
        AnnotationType,
        DatasetManifest,
        EvaluationScope,
        EvaluationUnit,
        EvaluationUnitType,
        Formant,
        IntervalType,
        PredictionRun,
        PredictionRunManifest,
        PreparedDataset,
        TrackingInputMode,
    )

__all__ = [
    "AnnotationType",
    "DatasetManifest",
    "EvaluationScope",
    "EvaluationUnit",
    "EvaluationUnitType",
    "Formant",
    "IntervalType",
    "PredictionRun",
    "PredictionRunManifest",
    "PreparedDataset",
    "TrackingInputMode",
]


def __getattr__(name: str) -> Any:
    """Load backwards-compatible public model exports only when requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from formant_benchmark.data import models

    value = getattr(models, name)
    globals()[name] = value
    return value
