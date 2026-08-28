"""Typed domain models shared across preparation, tracking, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Formant(StrEnum):
    """Canonical formant vocabulary used by all normalized representations."""

    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"


class AnnotationType(StrEnum):
    """Gold representation present in a prepared dataset."""

    TRACK = "track"
    STATIC = "static"
    MIXED = "mixed"


class IntervalType(StrEnum):
    """Common interval kinds reserved by the benchmark architecture."""

    PHONE = "phone"
    VOWEL = "vowel"
    VOICED = "voiced"
    WORD = "word"


class IntervalOrigin(StrEnum):
    """Provenance of a prepared temporal interval."""

    SOURCE = "source"
    DERIVED = "derived"
    BENCHMARK = "benchmark"


class EvaluationScope(StrEnum):
    """Evaluation scopes represented by V1; voiced is intentionally unavailable."""

    ALL = "all"
    VOWELS = "vowels"
    VOICED = "voiced"


class EvaluationUnitType(StrEnum):
    """Semantic kind of one independently scored evaluation unit."""

    INTERVAL = "interval"
    STATIC_MEASUREMENT = "static_measurement"


class TrackingInputMode(StrEnum):
    """Tracking input modes reserved by the architecture."""

    FULL_ITEM = "full_item"
    CROPPED_INTERVALS = "cropped_intervals"
    FULL_ITEM_WITH_INTERVALS = "full_item_with_intervals"


class DatasetManifest(BaseModel):
    """Persisted identity/configuration metadata for one prepared dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    adapter_version: str | None = None
    schema_version: str = "1"
    annotation_type: AnnotationType
    available_formants: list[Formant]
    preparation_config: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str | None = None

    @model_validator(mode="after")
    def validate_available_formants(self) -> "DatasetManifest":
        if not self.available_formants:
            raise ValueError("A prepared dataset must declare at least one available formant.")
        if len(set(self.available_formants)) != len(self.available_formants):
            raise ValueError("available_formants must not contain duplicates.")
        order = {formant: index for index, formant in enumerate(Formant)}
        self.available_formants = sorted(self.available_formants, key=order.__getitem__)
        return self


class EvaluationUnit(BaseModel):
    """One independently scored interval or source-defined static observation."""

    model_config = ConfigDict(extra="forbid")

    evaluation_unit_id: str = Field(min_length=1)
    evaluation_unit_type: EvaluationUnitType
    item_id: str = Field(min_length=1)
    scope: EvaluationScope
    interval_id: str | None = None
    measurement_id: str | None = None
    region: str = "full"
    start_s: float | None = None
    end_s: float | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> "EvaluationUnit":
        if self.evaluation_unit_type is EvaluationUnitType.STATIC_MEASUREMENT:
            if not self.measurement_id:
                raise ValueError("Static evaluation units require measurement_id.")
            if self.start_s is not None or self.end_s is not None:
                raise ValueError("Static evaluation units do not define interval bounds.")
        else:
            if self.scope is EvaluationScope.VOWELS and not self.interval_id:
                raise ValueError("Vowel interval evaluation units require interval_id.")
            if (self.start_s is None) != (self.end_s is None):
                raise ValueError("Interval bounds must be supplied together.")
            if self.start_s is not None and self.end_s is not None and self.end_s <= self.start_s:
                raise ValueError("Evaluation interval end_s must be greater than start_s.")
        return self


@dataclass(slots=True)
class PreparedDataset:
    """In-memory prepared dataset using the benchmark's normalized tables."""

    manifest: DatasetManifest
    items: pd.DataFrame
    tracks: pd.DataFrame
    intervals: pd.DataFrame
    splits: pd.DataFrame
    static_measurements: pd.DataFrame | None = None
    root: Path | None = None

    def with_root(self, root: Path) -> "PreparedDataset":
        """Return a shallow copy associated with a persisted dataset directory."""
        return replace(self, root=root)
