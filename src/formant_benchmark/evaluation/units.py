"""Deterministic construction of interval and static evaluation units."""

from __future__ import annotations

import pandas as pd

from formant_benchmark.data.models import (
    AnnotationType,
    EvaluationScope,
    EvaluationUnit,
    EvaluationUnitType,
    PreparedDataset,
)
from formant_benchmark.exceptions import EvaluationCompatibilityError


def build_evaluation_units(
    dataset: PreparedDataset,
    *,
    scope: EvaluationScope,
    item_ids: set[str],
    central_region: float | None = None,
) -> list[EvaluationUnit]:
    """Build independently scored units for selected prepared items."""
    if dataset.manifest.annotation_type is AnnotationType.MIXED:
        raise EvaluationCompatibilityError(
            "Mixed track/static evaluation is not implemented yet; prepare a track or static dataset separately."
        )
    if central_region is not None and scope is not EvaluationScope.VOWELS:
        raise EvaluationCompatibilityError("--central-region is only valid with vowel evaluation scope.")
    if central_region is not None and dataset.manifest.annotation_type is not AnnotationType.TRACK:
        raise EvaluationCompatibilityError("--central-region applies only to trajectory vowel evaluation.")

    if dataset.manifest.annotation_type is AnnotationType.STATIC:
        units = _static_units(dataset, scope=scope, item_ids=item_ids)
    else:
        units = _interval_units(
            dataset,
            scope=scope,
            item_ids=item_ids,
            central_region=central_region,
        )
    if not units:
        raise EvaluationCompatibilityError(
            f"No evaluable units are available for scope '{scope.value}' and the selected prediction inputs."
        )
    return units


def _interval_units(
    dataset: PreparedDataset,
    *,
    scope: EvaluationScope,
    item_ids: set[str],
    central_region: float | None,
) -> list[EvaluationUnit]:
    if scope is EvaluationScope.ALL:
        selected = dataset.items.loc[dataset.items["item_id"].astype(str).isin(item_ids)]
        return [
            EvaluationUnit(
                evaluation_unit_id=f"item:{row.item_id}:all",
                evaluation_unit_type=EvaluationUnitType.INTERVAL,
                item_id=str(row.item_id),
                scope=scope,
                start_s=0.0,
                end_s=float(row.duration_s),
            )
            for row in selected.sort_values("item_id", kind="stable").itertuples(index=False)
        ]

    vowels = dataset.intervals.loc[
        (dataset.intervals["interval_type"].astype(str) == "vowel")
        & dataset.intervals["item_id"].astype(str).isin(item_ids)
    ].copy()
    units: list[EvaluationUnit] = []
    for row in vowels.sort_values(["item_id", "start_s", "interval_id"], kind="stable").itertuples(index=False):
        start_s = float(row.start_s)
        end_s = float(row.end_s)
        units.append(
            EvaluationUnit(
                evaluation_unit_id=f"interval:{row.interval_id}:full",
                evaluation_unit_type=EvaluationUnitType.INTERVAL,
                item_id=str(row.item_id),
                scope=scope,
                interval_id=str(row.interval_id),
                region="full",
                start_s=start_s,
                end_s=end_s,
            )
        )
        if central_region is not None:
            center = (start_s + end_s) / 2.0
            half_width = (end_s - start_s) * central_region / 2.0
            units.append(
                EvaluationUnit(
                    evaluation_unit_id=f"interval:{row.interval_id}:central_{central_region:g}",
                    evaluation_unit_type=EvaluationUnitType.INTERVAL,
                    item_id=str(row.item_id),
                    scope=scope,
                    interval_id=str(row.interval_id),
                    region=f"central_{central_region:g}",
                    start_s=center - half_width,
                    end_s=center + half_width,
                )
            )
    return units


def _static_units(
    dataset: PreparedDataset,
    *,
    scope: EvaluationScope,
    item_ids: set[str],
) -> list[EvaluationUnit]:
    if dataset.static_measurements is None:
        return []
    measurements = dataset.static_measurements.loc[
        dataset.static_measurements["item_id"].astype(str).isin(item_ids)
    ].copy()
    if scope is EvaluationScope.VOWELS:
        vowel_ids = set(
            dataset.intervals.loc[
                dataset.intervals["interval_type"].astype(str) == "vowel",
                "interval_id",
            ].astype(str)
        )
        measurements = measurements.loc[
            measurements["interval_id"].notna()
            & measurements["interval_id"].astype(str).isin(vowel_ids)
        ]
    return [
        EvaluationUnit(
            evaluation_unit_id=f"static:{row.measurement_id}",
            evaluation_unit_type=EvaluationUnitType.STATIC_MEASUREMENT,
            item_id=str(row.item_id),
            scope=scope,
            interval_id=None if pd.isna(row.interval_id) else str(row.interval_id),
            measurement_id=str(row.measurement_id),
        )
        for row in measurements.sort_values(["item_id", "measurement_id"], kind="stable").itertuples(index=False)
    ]


__all__ = ["EvaluationUnit", "EvaluationUnitType", "build_evaluation_units"]
