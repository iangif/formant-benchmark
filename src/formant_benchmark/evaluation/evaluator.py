"""Core Phase 6 evaluation orchestration for trajectory and static gold."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from formant_benchmark.data.models import (
    AnnotationType,
    EvaluationResult,
    EvaluationRunManifest,
    EvaluationScope,
    EvaluationUnit,
    EvaluationUnitType,
    Formant,
    PredictionRun,
    PreparedDataset,
    TrackingInputMode,
)
from formant_benchmark.evaluation.aggregation import aggregate_unit_metrics
from formant_benchmark.evaluation.align import interpolate_formant
from formant_benchmark.evaluation.metrics import DEFAULT_METRICS, avg, coverage, fdr, mae, rmse
from formant_benchmark.evaluation.scopes import require_implemented_scope
from formant_benchmark.evaluation.static import static_prediction
from formant_benchmark.evaluation.units import build_evaluation_units
from formant_benchmark.exceptions import (
    DatasetFingerprintMismatchError,
    EvaluationCompatibilityError,
    IncompatibleFormantsError,
)
from formant_benchmark.preparation.fingerprint import dataset_fingerprint

_CANONICAL_FORMANTS = tuple(Formant)
_FORMANT_ORDER = {formant: index for index, formant in enumerate(_CANONICAL_FORMANTS)}
_METRIC_FUNCTIONS = {"rmse": rmse, "mae": mae, "avg": avg, "fdr": fdr, "coverage": coverage}
_TIME_TOLERANCE_S = 1e-9


def evaluate(
    dataset: PreparedDataset,
    prediction_run: PredictionRun,
    *,
    scope: str | EvaluationScope,
    formants: Sequence[str | Formant] | None = None,
    metrics: Sequence[str] | None = None,
    central_region: float | None = None,
    group_by: Sequence[Sequence[str]] = (),
    split: str | None = None,
    fdr_relative_threshold: float = 0.30,
    fdr_absolute_threshold_hz: float = 300.0,
    evaluation_id: str = "evaluation",
) -> EvaluationResult:
    """Evaluate one persisted prediction condition against one prepared dataset."""
    parsed_scope = require_implemented_scope(scope)
    _validate_numeric_options(
        central_region=central_region,
        fdr_relative_threshold=fdr_relative_threshold,
        fdr_absolute_threshold_hz=fdr_absolute_threshold_hz,
    )
    selected_metrics = _resolve_metrics(metrics)
    effective_split, item_ids = _validate_compatibility_and_select_items(
        dataset,
        prediction_run,
        scope=parsed_scope,
        split=split,
    )
    selected_formants = _resolve_formants(dataset, prediction_run, formants)
    units = build_evaluation_units(
        dataset,
        scope=parsed_scope,
        item_ids=item_ids,
        central_region=central_region,
    )

    detailed = _score_units(
        dataset,
        prediction_run,
        units,
        selected_formants=selected_formants,
        selected_metrics=selected_metrics,
        fdr_relative_threshold=fdr_relative_threshold,
        fdr_absolute_threshold_hz=fdr_absolute_threshold_hz,
    )
    metric_columns = _metric_output_columns(selected_metrics)
    aggregates = aggregate_unit_metrics(
        detailed,
        metric_columns=metric_columns,
        group_by=group_by,
    )
    numerical_columns = [column for column in metric_columns if column in detailed]
    numerical_units = 0
    if numerical_columns:
        numeric = detailed[numerical_columns].apply(pd.to_numeric, errors="coerce")
        numerical_units = int(np.isfinite(numeric.to_numpy(dtype=float)).any(axis=1).sum())

    fingerprint = dataset.manifest.fingerprint or dataset_fingerprint(dataset)
    manifest = EvaluationRunManifest(
        evaluation_id=evaluation_id,
        dataset_name=dataset.manifest.name,
        dataset_fingerprint=fingerprint,
        prediction_run_id=prediction_run.manifest.run_id,
        tracker=prediction_run.manifest.tracker,
        scope=parsed_scope,
        split=effective_split,
        selected_formants=selected_formants,
        metrics=selected_metrics,
        central_region=central_region,
        fdr_relative_threshold=fdr_relative_threshold,
        fdr_absolute_threshold_hz=fdr_absolute_threshold_hz,
        group_by=[list(group) for group in group_by],
        n_evaluation_units=len(detailed),
        n_units_with_numerical_metrics=numerical_units,
        created_at=_timestamp(),
        benchmark_version=_benchmark_version(),
        benchmark_commit=_benchmark_commit(),
    )
    return EvaluationResult(manifest, detailed, aggregates)


def _validate_numeric_options(
    *,
    central_region: float | None,
    fdr_relative_threshold: float,
    fdr_absolute_threshold_hz: float,
) -> None:
    if central_region is not None and not 0 < central_region <= 1:
        raise EvaluationCompatibilityError("central_region must be greater than 0 and at most 1.")
    if fdr_relative_threshold <= 0:
        raise EvaluationCompatibilityError("FDR relative-error threshold must be greater than zero.")
    if fdr_absolute_threshold_hz <= 0:
        raise EvaluationCompatibilityError("FDR absolute-error threshold must be greater than zero.")


def _resolve_metrics(metrics: Sequence[str] | None) -> list[str]:
    values = list(DEFAULT_METRICS if metrics is None else metrics)
    if not values:
        raise EvaluationCompatibilityError("At least one evaluation metric must be selected.")
    normalized: list[str] = []
    for raw in values:
        name = str(raw).lower()
        if name not in _METRIC_FUNCTIONS:
            raise EvaluationCompatibilityError(
                f"Unknown metric '{raw}'. Supported V1 metrics: {', '.join(DEFAULT_METRICS)}."
            )
        if name not in normalized:
            normalized.append(name)
    return normalized


def _resolve_formants(
    dataset: PreparedDataset,
    prediction_run: PredictionRun,
    requested: Sequence[str | Formant] | None,
) -> list[Formant]:
    gold = set(dataset.manifest.available_formants)
    available_predictions = set(prediction_run.manifest.prediction_formants)
    compatible = gold & available_predictions
    if requested is None:
        selected = sorted(compatible, key=_FORMANT_ORDER.__getitem__)
        if not selected:
            raise IncompatibleFormantsError(
                "Gold data and prediction run declare no compatible formants."
            )
        return selected

    try:
        parsed = [value if isinstance(value, Formant) else Formant(str(value).upper()) for value in requested]
    except ValueError as exc:
        raise IncompatibleFormantsError("Formants must use the canonical names F1, F2, F3, or F4.") from exc
    if not parsed:
        raise IncompatibleFormantsError("At least one formant must be selected for evaluation.")
    if len(set(parsed)) != len(parsed):
        raise IncompatibleFormantsError("Explicit formant selection must not contain duplicates.")
    unavailable_gold = [value.value for value in parsed if value not in gold]
    unavailable_prediction = [value.value for value in parsed if value not in available_predictions]
    if unavailable_gold or unavailable_prediction:
        details = []
        if unavailable_gold:
            details.append(f"not available in gold: {', '.join(unavailable_gold)}")
        if unavailable_prediction:
            details.append(f"not available in prediction run: {', '.join(unavailable_prediction)}")
        raise IncompatibleFormantsError("Requested formants are incompatible (" + "; ".join(details) + ").")
    return sorted(parsed, key=_FORMANT_ORDER.__getitem__)


def _validate_compatibility_and_select_items(
    dataset: PreparedDataset,
    prediction_run: PredictionRun,
    *,
    scope: EvaluationScope,
    split: str | None,
) -> tuple[str | None, set[str]]:
    fingerprint = dataset.manifest.fingerprint or dataset_fingerprint(dataset)
    if prediction_run.manifest.dataset_fingerprint != fingerprint:
        raise DatasetFingerprintMismatchError(
            "Prediction run was created from a different prepared dataset fingerprint: "
            f"run={prediction_run.manifest.dataset_fingerprint}, dataset={fingerprint}."
        )
    if (
        prediction_run.manifest.input_mode is TrackingInputMode.CROPPED_INTERVALS
        and scope is EvaluationScope.ALL
    ):
        interval_type = prediction_run.manifest.interval_type or "unspecified"
        raise EvaluationCompatibilityError(
            "Prediction run was generated from cropped intervals "
            f"('{interval_type}') and cannot be evaluated with scope 'all'. "
            "Use an interval-compatible evaluation scope or create a full_item prediction run."
        )
    if prediction_run.manifest.status != "completed":
        raise EvaluationCompatibilityError(
            f"Prediction run status is '{prediction_run.manifest.status}'; evaluation requires a completed run."
        )
    run_split = prediction_run.manifest.split
    if split is not None and run_split is not None and split != run_split:
        raise EvaluationCompatibilityError(
            f"Prediction run contains split '{run_split}', so it cannot be evaluated as split '{split}'."
        )
    effective_split = split if split is not None else run_split

    attempted_items = set(prediction_run.item_parameters["item_id"].astype(str))
    if not attempted_items:
        raise EvaluationCompatibilityError("Prediction run contains no attempted tracking inputs.")
    known_items = set(dataset.items["item_id"].astype(str))
    unknown = sorted(attempted_items - known_items)
    if unknown:
        raise EvaluationCompatibilityError(f"Prediction run references unknown item_id(s): {unknown[:10]}")

    selected_items = attempted_items
    if effective_split is not None:
        split_rows = dataset.splits.loc[dataset.splits["split"].astype(str) == effective_split]
        if split_rows.empty:
            raise EvaluationCompatibilityError(f"Prepared dataset contains no split '{effective_split}'.")
        split_items = set(split_rows["item_id"].astype(str))
        selected_items = attempted_items & split_items
        if not selected_items:
            raise EvaluationCompatibilityError(
                f"Prediction run contains no attempted items in split '{effective_split}'."
            )
    return effective_split, selected_items


def _score_units(
    dataset: PreparedDataset,
    prediction_run: PredictionRun,
    units: Sequence[EvaluationUnit],
    *,
    selected_formants: Sequence[Formant],
    selected_metrics: Sequence[str],
    fdr_relative_threshold: float,
    fdr_absolute_threshold_hz: float,
) -> pd.DataFrame:
    predictions = {
        str(item_id): frame.sort_values("time_s", kind="stable")
        for item_id, frame in prediction_run.predictions.groupby("item_id", sort=False)
    }
    empty_predictions = prediction_run.predictions.iloc[0:0].copy()
    tracks = {
        str(item_id): frame.sort_values("time_s", kind="stable")
        for item_id, frame in dataset.tracks.groupby("item_id", sort=False)
    }
    intervals = {
        str(row["interval_id"]): row
        for row in dataset.intervals.to_dict(orient="records")
    }
    measurements = (
        {
            str(row["measurement_id"]): row
            for row in dataset.static_measurements.to_dict(orient="records")
        }
        if dataset.static_measurements is not None
        else {}
    )
    items = {
        str(row["item_id"]): row
        for row in dataset.items.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    for unit in units:
        prediction = predictions.get(unit.item_id, empty_predictions)
        if unit.evaluation_unit_type is EvaluationUnitType.INTERVAL:
            gold, aligned = _trajectory_values(
                tracks.get(unit.item_id, dataset.tracks.iloc[0:0]),
                prediction,
                unit,
                selected_formants,
            )
        else:
            measurement = measurements[unit.measurement_id or ""]
            interval = intervals.get(str(measurement.get("interval_id"))) if pd.notna(measurement.get("interval_id")) else None
            gold, aligned = _static_values(
                measurement,
                prediction,
                selected_formants,
                interval=interval,
                item_duration_s=float(items[unit.item_id]["duration_s"]),
            )
        row = _identity_row(unit)
        row.update(_metadata_row(unit, items[unit.item_id], intervals, measurements))
        row.update(
            _metrics_row(
                gold,
                aligned,
                selected_formants=selected_formants,
                selected_metrics=selected_metrics,
                fdr_relative_threshold=fdr_relative_threshold,
                fdr_absolute_threshold_hz=fdr_absolute_threshold_hz,
            )
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    identity = [
        "evaluation_unit_id",
        "evaluation_unit_type",
        "item_id",
        "interval_id",
        "measurement_id",
        "scope",
        "region",
    ]
    metrics = _metric_output_columns(selected_metrics)
    counts = ["n_gold_measurements", "n_predicted_measurements"]
    metadata = [column for column in frame.columns if column not in {*identity, *metrics, *counts}]
    return frame.loc[:, [*identity, *metrics, *counts, *metadata]]


def _trajectory_values(
    gold_track: pd.DataFrame,
    prediction: pd.DataFrame,
    unit: EvaluationUnit,
    selected_formants: Sequence[Formant],
) -> tuple[dict[Formant, np.ndarray], dict[Formant, np.ndarray]]:
    if gold_track.empty:
        gold_times = np.asarray([], dtype=float)
        selected_gold = gold_track
    else:
        times = pd.to_numeric(gold_track["time_s"], errors="coerce")
        mask = (times >= float(unit.start_s) - _TIME_TOLERANCE_S) & (times <= float(unit.end_s) + _TIME_TOLERANCE_S)
        selected_gold = gold_track.loc[mask]
        gold_times = pd.to_numeric(selected_gold["time_s"], errors="coerce").to_numpy(dtype=float)
    gold = {
        formant: pd.to_numeric(selected_gold[formant.value], errors="coerce").to_numpy(dtype=float)
        for formant in selected_formants
    }
    aligned = {
        formant: interpolate_formant(prediction, gold_times, formant.value)
        for formant in selected_formants
    }
    return gold, aligned


def _static_values(
    measurement: dict[str, Any],
    prediction: pd.DataFrame,
    selected_formants: Sequence[Formant],
    *,
    interval: dict[str, Any] | None,
    item_duration_s: float,
) -> tuple[dict[Formant, np.ndarray], dict[Formant, np.ndarray]]:
    gold: dict[Formant, np.ndarray] = {}
    aligned: dict[Formant, np.ndarray] = {}
    for formant in selected_formants:
        gold_value = pd.to_numeric(pd.Series([measurement.get(formant.value)]), errors="coerce").iloc[0]
        gold[formant] = np.asarray([gold_value], dtype=float)
        predicted = static_prediction(
            measurement,
            prediction,
            formant=formant.value,
            interval=interval,
            item_duration_s=item_duration_s,
        )
        aligned[formant] = np.asarray([predicted], dtype=float)
    return gold, aligned


def _metrics_row(
    gold: dict[Formant, np.ndarray],
    prediction: dict[Formant, np.ndarray],
    *,
    selected_formants: Sequence[Formant],
    selected_metrics: Sequence[str],
    fdr_relative_threshold: float,
    fdr_absolute_threshold_hz: float,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    selected = set(selected_formants)
    for metric_name in selected_metrics:
        per_formant: list[float] = []
        for formant in _CANONICAL_FORMANTS:
            column = f"{metric_name}_{formant.value.lower()}"
            if formant not in selected:
                value = float("nan")
            elif metric_name == "fdr":
                value = fdr(
                    gold[formant],
                    prediction[formant],
                    relative_threshold=fdr_relative_threshold,
                    absolute_threshold_hz=fdr_absolute_threshold_hz,
                )
            else:
                value = float(_METRIC_FUNCTIONS[metric_name](gold[formant], prediction[formant]))
            row[column] = value
            if formant in selected and np.isfinite(value):
                per_formant.append(value)
        row[metric_name] = float(np.mean(per_formant)) if per_formant else float("nan")

    if selected_formants:
        lengths = [len(gold[formant]) for formant in selected_formants]
        n_positions = max(lengths, default=0)
        if n_positions:
            gold_matrix = np.column_stack([gold[formant] for formant in selected_formants])
            pred_matrix = np.column_stack([prediction[formant] for formant in selected_formants])
            eligible = np.isfinite(gold_matrix)
            row["n_gold_measurements"] = int(eligible.any(axis=1).sum())
            row["n_predicted_measurements"] = int((eligible & np.isfinite(pred_matrix)).any(axis=1).sum())
        else:
            row["n_gold_measurements"] = 0
            row["n_predicted_measurements"] = 0
    return row


def _identity_row(unit: EvaluationUnit) -> dict[str, Any]:
    return {
        "evaluation_unit_id": unit.evaluation_unit_id,
        "evaluation_unit_type": unit.evaluation_unit_type.value,
        "item_id": unit.item_id,
        "interval_id": unit.interval_id,
        "measurement_id": unit.measurement_id,
        "scope": unit.scope.value,
        "region": unit.region,
    }


def _metadata_row(
    unit: EvaluationUnit,
    item: dict[str, Any],
    intervals: dict[str, dict[str, Any]],
    measurements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    excluded_item = {"item_id", "audio_path"}
    metadata = {key: _clean_scalar(value) for key, value in item.items() if key not in excluded_item}
    if unit.interval_id and unit.interval_id in intervals:
        interval = intervals[unit.interval_id]
        metadata["interval_type"] = _clean_scalar(interval.get("interval_type"))
        metadata["interval_label"] = _clean_scalar(interval.get("label"))
        metadata["interval_origin"] = _clean_scalar(interval.get("origin"))
        if str(interval.get("interval_type")) == "vowel":
            metadata["vowel"] = _clean_scalar(interval.get("label"))
    if unit.measurement_id and unit.measurement_id in measurements:
        measurement = measurements[unit.measurement_id]
        excluded_measurement = {
            "measurement_id", "item_id", "interval_id", "relative_position", "time_s",
            "window_start_s", "window_end_s", "F1", "F2", "F3", "F4",
        }
        metadata.update(
            {key: _clean_scalar(value) for key, value in measurement.items() if key not in excluded_measurement}
        )
    return metadata


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _metric_output_columns(metrics: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for metric in metrics:
        columns.extend(f"{metric}_{formant.value.lower()}" for formant in _CANONICAL_FORMANTS)
        columns.append(metric)
    return columns


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _benchmark_version() -> str | None:
    try:
        return version("formant-benchmark")
    except PackageNotFoundError:
        return None


def _benchmark_commit() -> str | None:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None
