"""Overall descriptive summary of canonical evaluation-unit metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def aggregate_unit_metrics(detailed: pd.DataFrame, *, metric_columns: Sequence[str]) -> pd.DataFrame:
    """Average each unit-level metric over all evaluation units, per scope and region.

    This is a quick descriptive summary: an unweighted mean over evaluation units,
    ignoring units where a metric is undefined. Subgroup breakdowns and any
    speaker/token weighting are left to downstream analysis of the unit-level
    table, which carries the item and interval metadata needed for grouping.
    """
    rows: list[dict[str, object]] = []
    regions = detailed[["scope", "region"]].drop_duplicates().to_dict(orient="records")
    for region_key in regions:
        frame = detailed.loc[
            (detailed["scope"] == region_key["scope"]) & (detailed["region"] == region_key["region"])
        ]
        row: dict[str, object] = {
            "scope": str(region_key["scope"]),
            "region": str(region_key["region"]),
            "n_evaluation_units": int(len(frame)),
        }
        for column in metric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[column] = float(np.mean(finite)) if len(finite) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=["scope", "region", "n_evaluation_units", *metric_columns])
