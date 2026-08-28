"""Tests for constrained core domain models."""

import pytest
from pydantic import ValidationError

from formant_benchmark.data.models import (
    EvaluationScope,
    EvaluationUnit,
    EvaluationUnitType,
)


def test_vowel_interval_unit_requires_interval_id() -> None:
    with pytest.raises(ValidationError):
        EvaluationUnit(
            evaluation_unit_id="unit-1",
            evaluation_unit_type=EvaluationUnitType.INTERVAL,
            item_id="item-1",
            scope=EvaluationScope.VOWELS,
        )


def test_static_unit_requires_measurement_id() -> None:
    unit = EvaluationUnit(
        evaluation_unit_id="unit-2",
        evaluation_unit_type=EvaluationUnitType.STATIC_MEASUREMENT,
        item_id="item-1",
        measurement_id="measurement-1",
        scope=EvaluationScope.ALL,
    )
    assert unit.measurement_id == "measurement-1"


def test_available_formants_are_canonicalized_and_non_empty() -> None:
    from formant_benchmark.data.models import AnnotationType, DatasetManifest, Formant

    manifest = DatasetManifest(
        name="test",
        source="test",
        adapter="test",
        annotation_type=AnnotationType.TRACK,
        available_formants=[Formant.F4, Formant.F2],
    )
    assert manifest.available_formants == [Formant.F2, Formant.F4]

    with pytest.raises(ValidationError):
        DatasetManifest(
            name="test",
            source="test",
            adapter="test",
            annotation_type=AnnotationType.TRACK,
            available_formants=[],
        )
