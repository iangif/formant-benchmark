"""Simple descriptive aggregation of canonical evaluation-unit metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from formant_benchmark.exceptions import EvaluationCompatibilityError


def aggregate_unit_metrics(
    detailed: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    group_by: Sequence[Sequence[str]],
) -> pd.DataFrame:
    """Average unit-level metrics overall and for requested metadata groupings.

    V1 deliberately treats these summaries as descriptive conveniences. The
    evaluation-unit table remains canonical so later speaker/token weighting can
    be changed without rerunning trackers.
    """
    normalized_groups = _validate_groupings(detailed, group_by, metric_columns)
    metadata_columns = list(dict.fromkeys(column for group in normalized_groups for column in group))
    rows: list[dict[str, object]] = []
    regions = detailed[["scope", "region"]].drop_duplicates().to_dict(orient="records")

    for region_key in regions:
        region_frame = detailed.loc[
            (detailed["scope"] == region_key["scope"]) & (detailed["region"] == region_key["region"])
        ]
        rows.append(
            _summary_row(
                region_frame,
                grouping="overall",
                group_values={},
                metadata_columns=metadata_columns,
                metric_columns=metric_columns,
                scope=str(region_key["scope"]),
                region=str(region_key["region"]),
            )
        )
        for grouping in normalized_groups:
            group_key: str | list[str] = grouping[0] if len(grouping) == 1 else list(grouping)
            grouped = region_frame.groupby(group_key, dropna=False, sort=True)
            for key, frame in grouped:
                values = key if isinstance(key, tuple) else (key,)
                rows.append(
                    _summary_row(
                        frame,
                        grouping="+".join(grouping),
                        group_values=dict(zip(grouping, values, strict=True)),
                        metadata_columns=metadata_columns,
                        metric_columns=metric_columns,
                        scope=str(region_key["scope"]),
                        region=str(region_key["region"]),
                    )
                )

    columns = ["grouping", "scope", "region", "n_evaluation_units", *metadata_columns, *metric_columns]
    return pd.DataFrame(rows, columns=columns)


def _validate_groupings(
    detailed: pd.DataFrame,
    group_by: Sequence[Sequence[str]],
    metric_columns: Sequence[str],
) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    forbidden = set(metric_columns) | {"evaluation_unit_id", "n_gold_measurements", "n_predicted_measurements"}
    for raw_group in group_by:
        group = tuple(str(column).strip() for column in raw_group if str(column).strip())
        if not group:
            raise EvaluationCompatibilityError("Each subgroup specification must contain at least one metadata field.")
        if len(set(group)) != len(group):
            raise EvaluationCompatibilityError(f"Subgroup contains duplicate metadata fields: {group}")
        missing = [column for column in group if column not in detailed.columns]
        if missing:
            raise EvaluationCompatibilityError(
                f"Unknown subgroup metadata column(s): {', '.join(missing)}."
            )
        invalid = [column for column in group if column in forbidden]
        if invalid:
            raise EvaluationCompatibilityError(
                f"Cannot group by metric/identity column(s): {', '.join(invalid)}."
            )
        if group not in result:
            result.append(group)
    return result


def _summary_row(
    frame: pd.DataFrame,
    *,
    grouping: str,
    group_values: dict[str, object],
    metadata_columns: list[str],
    metric_columns: Sequence[str],
    scope: str,
    region: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "grouping": grouping,
        "scope": scope,
        "region": region,
        "n_evaluation_units": int(len(frame)),
    }
    for column in metadata_columns:
        row[column] = group_values.get(column, None)
    for column in metric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        row[column] = float(np.mean(finite)) if len(finite) else float("nan")
    return row
