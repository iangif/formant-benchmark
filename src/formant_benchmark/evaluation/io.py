"""Persistence and structural validation for Phase 6 evaluation artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from formant_benchmark.data.models import EvaluationResult, EvaluationRunManifest, Formant
from formant_benchmark.exceptions import EvaluationAlreadyExistsError, EvaluationValidationError

_UNIT_IDENTITY_COLUMNS = (
    "evaluation_unit_id",
    "evaluation_unit_type",
    "item_id",
    "interval_id",
    "measurement_id",
    "scope",
    "region",
)
_UNIT_COUNT_COLUMNS = ("n_gold_measurements", "n_predicted_measurements")
_AGGREGATE_IDENTITY_COLUMNS = ("grouping", "scope", "region", "n_evaluation_units")


def write_evaluation_result(
    result: EvaluationResult,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> EvaluationResult:
    """Validate and atomically persist a complete evaluation directory."""
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        raise EvaluationAlreadyExistsError(f"Evaluation destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    validate_evaluation_result(result)

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.tmp-", dir=destination_path.parent))
    backup: Path | None = None
    try:
        _write_directory(result, temporary)
        load_evaluation_result(temporary)
        if destination_path.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{destination_path.name}.backup-", dir=destination_path.parent))
            backup.rmdir()
            os.replace(destination_path, backup)
        os.replace(temporary, destination_path)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if backup is not None and backup.exists() and not destination_path.exists():
            os.replace(backup, destination_path)
        raise
    result.root = destination_path
    return result


def load_evaluation_result(path: str | Path) -> EvaluationResult:
    """Load and structurally validate a persisted evaluation result."""
    root = Path(path)
    try:
        manifest = EvaluationRunManifest.model_validate(
            yaml.safe_load((root / "evaluation_manifest.yaml").read_text(encoding="utf-8"))
        )
        result = EvaluationResult(
            manifest=manifest,
            evaluation_unit_metrics=pd.read_parquet(root / "evaluation_unit_metrics.parquet"),
            aggregate_metrics=pd.read_parquet(root / "aggregate_metrics.parquet"),
            root=root,
        )
    except (OSError, ValueError, TypeError, ImportError) as exc:
        raise EvaluationValidationError(f"Could not load evaluation result '{root}': {exc}") from exc
    validate_evaluation_result(result)
    return result


def validate_evaluation_result(result: EvaluationResult) -> None:
    """Validate manifest/table agreement and required wide-format columns."""
    detailed = result.evaluation_unit_metrics
    aggregate = result.aggregate_metrics
    metric_columns = _metric_columns(result.manifest.metrics)
    _require_columns(detailed, (*_UNIT_IDENTITY_COLUMNS, *metric_columns, *_UNIT_COUNT_COLUMNS), "evaluation_unit_metrics.parquet")
    _require_columns(aggregate, (*_AGGREGATE_IDENTITY_COLUMNS, *metric_columns), "aggregate_metrics.parquet")
    if detailed["evaluation_unit_id"].isna().any() or detailed["evaluation_unit_id"].duplicated().any():
        raise EvaluationValidationError("evaluation_unit_id values must be non-null and unique.")
    if len(detailed) != result.manifest.n_evaluation_units:
        raise EvaluationValidationError(
            "Evaluation manifest n_evaluation_units does not match evaluation_unit_metrics.parquet."
        )
    if not detailed.empty:
        if set(detailed["scope"].astype(str)) != {result.manifest.scope.value}:
            raise EvaluationValidationError("Detailed evaluation scope does not match evaluation manifest.")
        valid_types = {"interval", "static_measurement"}
        unknown_types = set(detailed["evaluation_unit_type"].astype(str)) - valid_types
        if unknown_types:
            raise EvaluationValidationError(f"Unknown evaluation_unit_type values: {sorted(unknown_types)}")
    for count_column in _UNIT_COUNT_COLUMNS:
        values = pd.to_numeric(detailed[count_column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise EvaluationValidationError(f"{count_column} must contain non-negative counts.")


def inspect_evaluation_result(result: EvaluationResult) -> dict[str, object]:
    """Return a concise CLI summary of a completed evaluation."""
    return {
        "evaluation_id": result.manifest.evaluation_id,
        "dataset": result.manifest.dataset_name,
        "prediction_run": result.manifest.prediction_run_id,
        "tracker": result.manifest.tracker,
        "scope": result.manifest.scope.value,
        "split": result.manifest.split,
        "selected_formants": [formant.value for formant in result.manifest.selected_formants],
        "metrics": list(result.manifest.metrics),
        "central_region": result.manifest.central_region,
        "n_evaluation_units": result.manifest.n_evaluation_units,
        "n_units_with_numerical_metrics": result.manifest.n_units_with_numerical_metrics,
        "n_aggregate_rows": len(result.aggregate_metrics),
    }


def _write_directory(result: EvaluationResult, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "evaluation_manifest.yaml").write_text(
        yaml.safe_dump(result.manifest.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    result.evaluation_unit_metrics.to_parquet(root / "evaluation_unit_metrics.parquet", index=False)
    result.aggregate_metrics.to_parquet(root / "aggregate_metrics.parquet", index=False)


def _metric_columns(metrics: list[str]) -> tuple[str, ...]:
    columns: list[str] = []
    for metric in metrics:
        columns.extend(f"{metric}_{formant.value.lower()}" for formant in Formant)
        columns.append(metric)
    return tuple(columns)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EvaluationValidationError(f"{name} is missing columns: {', '.join(missing)}")
