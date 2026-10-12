"""Prediction-run artifact persistence and structural validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from formant_benchmark.data.models import PredictionRun, PredictionRunManifest
from formant_benchmark.data.schemas import (
    FAILURE_COLUMNS,
    ITEM_PARAMETER_COLUMNS,
    PREDICTION_COLUMNS,
)
from formant_benchmark.exceptions import PredictionRunValidationError

_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def write_prediction_run(run: PredictionRun, destination: str | Path) -> PredictionRun:
    """Checkpoint all four run artifacts using per-file atomic replacement."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    validate_prediction_run(run)
    _atomic_text(root / "run_manifest.yaml", yaml.safe_dump(run.manifest.model_dump(mode="json"), sort_keys=False))
    _atomic_parquet(root / "predictions.parquet", run.predictions.loc[:, PREDICTION_COLUMNS])
    _atomic_parquet(root / "failures.parquet", run.failures.loc[:, FAILURE_COLUMNS])
    _atomic_parquet(root / "item_parameters.parquet", run.item_parameters.loc[:, ITEM_PARAMETER_COLUMNS])
    run.root = root
    return run


def read_prediction_run_manifest(path: str | Path) -> PredictionRunManifest:
    """Read only a run's manifest, e.g. to compare digests without loading predictions."""
    manifest_path = Path(path) / "run_manifest.yaml"
    try:
        # Manifests list every input unit id, so the C loader matters for large runs.
        raw = yaml.load(manifest_path.read_text(encoding="utf-8"), Loader=_YAML_LOADER)
        return PredictionRunManifest.model_validate(raw)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise PredictionRunValidationError(f"Could not read run manifest '{manifest_path}': {exc}") from exc


def load_prediction_run(path: str | Path) -> PredictionRun:
    """Load and validate a persisted prediction run."""
    root = Path(path)
    try:
        run = PredictionRun(
            manifest=PredictionRunManifest.model_validate(
                yaml.safe_load((root / "run_manifest.yaml").read_text(encoding="utf-8"))
            ),
            predictions=pd.read_parquet(root / "predictions.parquet"),
            failures=pd.read_parquet(root / "failures.parquet"),
            item_parameters=pd.read_parquet(root / "item_parameters.parquet"),
            root=root,
        )
    except (OSError, ValueError, TypeError, ImportError) as exc:
        raise PredictionRunValidationError(f"Could not load prediction run '{root}': {exc}") from exc
    validate_prediction_run(run)
    return run


def validate_prediction_run(run: PredictionRun) -> None:
    """Validate schemas, normalized coordinates, and input identities."""
    _require_columns(run.predictions, PREDICTION_COLUMNS, "predictions.parquet")
    _require_columns(run.failures, FAILURE_COLUMNS, "failures.parquet")
    _require_columns(run.item_parameters, ITEM_PARAMETER_COLUMNS, "item_parameters.parquet")
    if run.item_parameters["input_unit_id"].duplicated().any():
        raise PredictionRunValidationError("item_parameters input_unit_id values must be unique.")
    if not run.predictions.empty:
        times = pd.to_numeric(run.predictions["time_s"], errors="coerce")
        if times.isna().any() or (times < 0).any() or not np.isfinite(times.to_numpy(dtype=float)).all():
            raise PredictionRunValidationError("Prediction time_s values must be finite and non-negative.")
        if run.predictions.duplicated(["item_id", "time_s"]).any():
            raise PredictionRunValidationError("predictions.parquet contains duplicate (item_id, time_s) rows.")
        for formant in ("F1", "F2", "F3", "F4"):
            values = pd.to_numeric(run.predictions[formant], errors="coerce")
            if (run.predictions[formant].notna() & values.isna()).any():
                raise PredictionRunValidationError(f"Prediction {formant} values must be numeric or missing.")


def inspect_prediction_run(run: PredictionRun) -> dict[str, object]:
    """Return a concise CLI summary."""
    return {
        "run_id": run.manifest.run_id,
        "status": run.manifest.status,
        "dataset": run.manifest.dataset_name,
        "tracker": run.manifest.tracker,
        "input_mode": run.manifest.input_mode.value,
        "interval_type": run.manifest.interval_type,
        "interval_padding_s": run.manifest.interval_padding_s,
        "split": run.manifest.split,
        "prediction_formants": [value.value for value in run.manifest.prediction_formants],
        "requested_inputs": run.manifest.requested_inputs,
        "succeeded_inputs": run.manifest.succeeded_inputs,
        "failed_inputs": run.manifest.failed_inputs,
        "prediction_rows": len(run.predictions),
    }


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise PredictionRunValidationError(f"{name} is missing columns: {', '.join(missing)}")


def _atomic_text(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
