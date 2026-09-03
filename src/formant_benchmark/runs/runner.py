"""Generic continue-on-error and resumable tracker-run orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from formant_benchmark.config.resolution import deep_merge, resolve_parameters
from formant_benchmark.data.models import (
    Formant,
    PredictionRun,
    PredictionRunManifest,
    TrackingInputMode,
)
from formant_benchmark.data.schemas import (
    empty_failures,
    empty_item_parameters,
    empty_predictions,
)
from formant_benchmark.exceptions import (
    ConfigurationError,
    PredictionRunAlreadyExistsError,
    ResumeCompatibilityError,
    TrackerExecutionError,
)
from formant_benchmark.execution.backends import ExecutionWorker, backend_from_config
from formant_benchmark.runs.io import load_prediction_run, write_prediction_run
from formant_benchmark.tracker_wrappers.protocol import normalize_wrapper_rows
from formant_benchmark.trackers.base import TrackerAdapter, TrackingInput


def execute_prediction_run(
    *,
    tracker: TrackerAdapter,
    inputs: Sequence[TrackingInput],
    tracker_config: Mapping[str, Any],
    destination: str | Path,
    dataset_fingerprint: str,
    dataset_name: str,
    input_mode: TrackingInputMode,
    dataset_tracker_config: Mapping[str, Any] | None = None,
    experiment_config: Mapping[str, Any] | None = None,
    cli_parameters: Mapping[str, Any] | None = None,
    interval_type: str | None = None,
    split: str | None = None,
    resume: bool = False,
    fail_fast: bool = False,
    show_progress: bool = False,
) -> PredictionRun:
    """Execute every pending input and checkpoint normalized artifacts."""
    root = Path(destination)
    effective_config = deep_merge(tracker.default_configuration, tracker_config)
    command = tuple(tracker.wrapper_command(effective_config))
    backend = backend_from_config(effective_config)
    check = backend.check(command)
    if not check["available"]:
        raise TrackerExecutionError("; ".join(check["problems"]))
    timeout_s = _timeout(effective_config)
    checkpoint_every = _execution_positive_int(effective_config, "checkpoint_every", default=50)
    max_items_per_process = _execution_positive_int(
        effective_config,
        "max_items_per_process",
        default=None,
    )

    compatibility = {
        "dataset_fingerprint": dataset_fingerprint,
        "tracker": tracker.name,
        "tracker_version": tracker.version,
        "tracker_config": effective_config,
        "dataset_tracker_config": dict(dataset_tracker_config or {}),
        "experiment_config": dict(experiment_config or {}),
        "cli_parameters": dict(cli_parameters or {}),
        "input_mode": input_mode.value,
        "interval_type": interval_type,
        "split": split,
        "input_unit_ids": [value.input_unit_id for value in inputs],
    }
    digest = _digest(compatibility)

    if len({value.input_unit_id for value in inputs}) != len(inputs):
        raise TrackerExecutionError("Tracking input_unit_id values must be unique within a run.")
    parameter_layers = (
        tracker.default_configuration,
        tracker_config,
        dataset_tracker_config,
        experiment_config,
        {"parameters": dict(cli_parameters or {})},
    )
    resolved_parameters: dict[str, dict[str, Any]] = {}
    for tracking_input in inputs:
        parameters = resolve_parameters(tracking_input.metadata, *parameter_layers)
        tracker.validate_parameters(parameters)
        resolved_parameters[tracking_input.input_unit_id] = parameters

    if root.exists():
        if not resume:
            raise PredictionRunAlreadyExistsError(
                f"Prediction run destination already exists: {root}. Use --resume explicitly to continue it."
            )
        run = load_prediction_run(root)
        if run.manifest.configuration_digest != digest:
            raise ResumeCompatibilityError(
                "Resume settings do not match the existing run; dataset, tracker, parameters, input mode, "
                "interval type, split, and input selection must be identical."
            )
        run.manifest.status = "running"
        run.manifest.completed_at = None
    else:
        manifest = PredictionRunManifest(
            run_id=root.name,
            status="running",
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            tracker=tracker.name,
            tracker_version=tracker.version,
            tracker_formants=_ordered_formants(tracker.capabilities.formants),
            input_mode=input_mode,
            interval_type=interval_type,
            split=split,
            configuration_digest=digest,
            configuration=_redact_configuration(compatibility),
            requested_inputs=len(inputs),
            created_at=_timestamp(),
        )
        run = PredictionRun(manifest, empty_predictions(), empty_failures(), empty_item_parameters(), root)
        write_prediction_run(run, root)

    completed = set(run.item_parameters["input_unit_id"].astype(str))
    supported = {formant.value for formant in tracker.capabilities.formants}
    prediction_keys = set(
        zip(run.predictions["item_id"].astype(str), run.predictions["time_s"].astype(float), strict=True)
    )
    pending = _PendingResults()

    with tempfile.TemporaryDirectory(prefix="formant-benchmark-run-") as temporary:
        temporary_root = Path(temporary)
        worker: ExecutionWorker | None = None
        worker_items = 0
        progress = tqdm(
            total=len(inputs),
            initial=sum(value.input_unit_id in completed for value in inputs),
            desc=f"Tracking {tracker.name}",
            unit="input",
            dynamic_ncols=True,
            disable=None if show_progress else True,
        )
        try:
            for tracking_input in inputs:
                if tracking_input.input_unit_id in completed:
                    continue
                if (
                    worker is None
                    or not worker.is_alive
                    or (max_items_per_process is not None and worker_items >= max_items_per_process)
                ):
                    if worker is not None:
                        worker.close()
                    worker = backend.start_worker(command, temporary_root)
                    worker_items = 0

                parameters = resolved_parameters[tracking_input.input_unit_id]
                parameter_row = {
                    "item_id": tracking_input.item_id,
                    "input_unit_id": tracking_input.input_unit_id,
                    "parameters_json": json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str),
                }
                try:
                    prediction = _execute_one(
                        tracker,
                        tracking_input,
                        parameters,
                        backend,
                        worker,
                        temporary_root,
                        timeout_s,
                        supported,
                    )
                    prediction["time_s"] = prediction["time_s"] + tracking_input.source_start_s
                    prediction = _deduplicate_predictions(prediction, prediction_keys)
                    if not prediction.empty:
                        pending.predictions.append(prediction)
                except Exception as exc:
                    failure_type = _failure_type(exc)
                    pending.failures.append({
                        "item_id": tracking_input.item_id,
                        "input_unit_id": tracking_input.input_unit_id,
                        "stage": "execution" if failure_type in {"execution_failed", "timeout"} else "output",
                        "failure_type": failure_type,
                        "message": str(exc),
                    })
                    if fail_fast:
                        pending.parameters.append(parameter_row)
                        pending.attempts += 1
                        _checkpoint(run, pending, root)
                        progress.update(1)
                        raise
                worker_items += 1
                if worker is not None and not worker.is_alive:
                    worker_items = 0
                pending.parameters.append(parameter_row)
                pending.attempts += 1
                if pending.attempts >= checkpoint_every:
                    _checkpoint(run, pending, root)
                progress.update(1)
        except BaseException:
            if pending.attempts:
                _checkpoint(run, pending, root)
            raise
        finally:
            if worker is not None:
                worker.close()
            progress.close()

    if pending.attempts:
        _checkpoint(run, pending, root)

    run.manifest.status = "completed"
    run.manifest.completed_at = _timestamp()
    _refresh_manifest(run)
    return write_prediction_run(run, root)


def _execute_one(
    tracker: TrackerAdapter,
    tracking_input: TrackingInput,
    parameters: Mapping[str, Any],
    backend: Any,
    worker: ExecutionWorker,
    temporary_root: Path,
    timeout_s: float | None,
    supported_formants: set[str],
) -> pd.DataFrame:
    unit_root = temporary_root / hashlib.sha256(tracking_input.input_unit_id.encode()).hexdigest()[:16]
    unit_root.mkdir(parents=True, exist_ok=True)
    audio_path: str | None = None
    if tracker.capabilities.requires_audio:
        wrapper_audio = tracking_input.audio_path
        if backend.stages_inputs:
            wrapper_audio = _stage_audio(tracking_input.audio_path, unit_root)
        audio_path = backend.protocol_path(wrapper_audio, temporary_root)
    result = worker.request(
        {
            "type": "request",
            "protocol_version": "2",
            "tracker": tracker.name,
            "item_id": tracking_input.item_id,
            "input_unit_id": tracking_input.input_unit_id,
            "audio_path": audio_path,
            "duration_s": tracking_input.duration_s,
            "parameters": dict(parameters),
            "metadata": dict(tracking_input.metadata),
            "intervals": list(tracking_input.intervals),
        },
        timeout_s=timeout_s,
    )
    if result.timed_out:
        raise TimeoutError(f"Tracker wrapper timed out after {timeout_s}s.")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise TrackerExecutionError(f"Tracker wrapper process failed: {detail[-1000:]}")
    response = result.response
    if response is None:
        detail = result.stdout.strip() or "missing JSON response"
        raise TrackerExecutionError(f"Tracker wrapper returned an invalid response: {detail[-1000:]}")
    if response.get("input_unit_id") != tracking_input.input_unit_id:
        raise TrackerExecutionError("Tracker wrapper response contains an unexpected input_unit_id.")
    if response.get("type") == "failure":
        detail = str(response.get("message") or response.get("error_type") or "unknown failure")
        raise TrackerExecutionError(f"Tracker wrapper failed: {detail}")
    if response.get("type") != "result" or not isinstance(response.get("rows"), list):
        raise TrackerExecutionError("Tracker wrapper returned an invalid result response.")
    return normalize_wrapper_rows(
        response["rows"],
        item_id=tracking_input.item_id,
        duration_s=tracking_input.duration_s,
        supported_formants=supported_formants,
    )


@dataclass(slots=True)
class _PendingResults:
    """Results accumulated between durable run checkpoints."""

    predictions: list[pd.DataFrame] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    attempts: int = 0


def _checkpoint(run: PredictionRun, pending: _PendingResults, root: Path) -> None:
    """Merge pending results once and atomically rewrite the canonical artifacts."""
    if pending.predictions:
        frames = ([run.predictions] if not run.predictions.empty else []) + pending.predictions
        run.predictions = pd.concat(frames, ignore_index=True)
    if pending.failures:
        run.failures = pd.concat([run.failures, pd.DataFrame(pending.failures)], ignore_index=True)
    if pending.parameters:
        run.item_parameters = pd.concat(
            [run.item_parameters, pd.DataFrame(pending.parameters)],
            ignore_index=True,
        )
    pending.predictions.clear()
    pending.failures.clear()
    pending.parameters.clear()
    pending.attempts = 0
    _refresh_manifest(run)
    write_prediction_run(run, root)


def _refresh_manifest(run: PredictionRun) -> None:
    failed = set(run.failures["input_unit_id"].astype(str))
    attempted = set(run.item_parameters["input_unit_id"].astype(str))
    run.manifest.failed_inputs = len(failed)
    run.manifest.succeeded_inputs = len(attempted - failed)
    run.manifest.prediction_formants = [
        formant for formant in run.manifest.tracker_formants
        if formant.value in run.predictions and run.predictions[formant.value].notna().any()
    ]


def _deduplicate_predictions(
    new: pd.DataFrame,
    keys: set[tuple[str, float]],
) -> pd.DataFrame:
    """Remove shared-boundary frames while updating one run-level key set."""
    keep = [
        (str(item_id), float(time_s)) not in keys
        for item_id, time_s in zip(new["item_id"], new["time_s"], strict=True)
    ]
    kept = new.loc[keep].copy()
    keys.update(
        zip(kept["item_id"].astype(str), kept["time_s"].astype(float), strict=True)
    )
    return kept


def _stage_audio(audio_path: Path, unit_root: Path) -> Path:
    """Make audio visible inside the backend work directory without copying when possible."""
    if not audio_path.is_file():
        raise TrackerExecutionError(f"Tracker audio file does not exist: {audio_path}")
    destination = unit_root / "input.wav"
    try:
        os.link(audio_path, destination)
    except OSError:
        shutil.copy2(audio_path, destination)
    return destination


def _failure_type(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, TrackerExecutionError):
        message = str(exc).lower()
        if "did not create" in message or "no prediction rows" in message:
            return "missing_output"
        if "wrapper failed" in message or "wrapper process failed" in message or "could not start" in message:
            return "execution_failed"
        return "invalid_output"
    return "execution_failed"


def _timeout(config: Mapping[str, Any]) -> float | None:
    execution = config.get("execution", {})
    value = execution.get("timeout_s") if isinstance(execution, Mapping) else None
    if value is None:
        return None
    timeout = float(value)
    if timeout <= 0:
        raise TrackerExecutionError("execution.timeout_s must be greater than zero.")
    return timeout


def _execution_positive_int(
    config: Mapping[str, Any],
    key: str,
    *,
    default: int | None,
) -> int | None:
    execution = config.get("execution", {})
    value = execution.get(key, default) if isinstance(execution, Mapping) else default
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigurationError(f"execution.{key} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"execution.{key} must be a positive integer.") from exc
    if parsed <= 0 or parsed != value:
        raise ConfigurationError(f"execution.{key} must be a positive integer.")
    return parsed


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _redact_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep run manifests reproducible without persisting environment secrets."""
    copied = json.loads(json.dumps(value, default=str))
    tracker_config = copied.get("tracker_config", {})
    execution = tracker_config.get("execution", {}) if isinstance(tracker_config, dict) else {}
    if isinstance(execution, dict) and "environment" in execution:
        environment = execution["environment"]
        if isinstance(environment, dict):
            execution["environment"] = {key: "<redacted>" for key in environment}
    return copied


def _ordered_formants(values: set[Formant] | frozenset[Formant]) -> list[Formant]:
    return [formant for formant in Formant if formant in values]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
