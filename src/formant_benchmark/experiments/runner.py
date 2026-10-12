"""Run one experiment: reuse or create its prediction run, then reuse or write its evaluation.

Tracking is the expensive step, so a run is identified by its tracking-condition
digest rather than by experiment name. Any completed run in the runs directory with
the same digest is reused; this is how several experiments that differ only in
evaluation settings share one set of predictions.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formant_benchmark.config.discovery import EVALUATIONS_ROOT, RUNS_ROOT
from formant_benchmark.data.models import (
    EvaluationResult,
    PredictionRun,
    PredictionRunManifest,
    PreparedDataset,
)
from formant_benchmark.evaluation import (
    evaluate,
    load_evaluation_result,
    read_evaluation_manifest,
    write_evaluation_result,
)
from formant_benchmark.evaluation.evaluator import evaluation_settings_digest
from formant_benchmark.evaluation.scopes import require_implemented_scope
from formant_benchmark.exceptions import (
    EvaluationValidationError,
    ExperimentConflictError,
    PredictionRunValidationError,
)
from formant_benchmark.execution.inputs import build_tracking_inputs
from formant_benchmark.experiments.config import ExperimentConfig
from formant_benchmark.preparation.fingerprint import dataset_fingerprint
from formant_benchmark.runs import load_prediction_run, read_prediction_run_manifest
from formant_benchmark.runs.runner import tracking_condition
from formant_benchmark.trackers import TRACKER_REGISTRY, register_builtin_trackers

_COMPLETED = "completed"


@dataclass(slots=True)
class ExperimentOutcome:
    """What an experiment run produced and whether each step did new work."""

    run: PredictionRun
    run_action: str  # tracked | resumed | reused
    evaluation: EvaluationResult
    evaluation_action: str  # evaluated | overwritten | reused


def run_experiment(
    experiment: ExperimentConfig,
    *,
    dataset: PreparedDataset,
    tracker_config: Mapping[str, Any],
    runs_root: Path = RUNS_ROOT,
    evaluations_root: Path = EVALUATIONS_ROOT,
    cli_parameters: Mapping[str, Any] | None = None,
    resume: bool = False,
    overwrite: bool = False,
    fail_fast: bool = False,
    show_progress: bool = False,
) -> ExperimentOutcome:
    """Track (or reuse matching predictions) into ``runs/<name>``, then evaluate into ``evaluations/<name>``.

    Fails before any tracking when the evaluation scope is unimplemented, when
    ``runs/<name>`` holds a different tracking condition, or when the matching run
    is incomplete and *resume* is false. An existing evaluation with the same
    settings is reused; one with different settings is replaced only with *overwrite*.
    """
    require_implemented_scope(experiment.evaluation.scope)
    register_builtin_trackers()
    tracker = TRACKER_REGISTRY.get(experiment.tracker)()
    fingerprint = dataset.manifest.fingerprint or dataset_fingerprint(dataset)
    dataset_tracker_config = dataset.manifest.tracker_overrides.get(tracker.name)
    tracking = experiment.tracking
    experiment_layer = tracking.parameter_layer()

    with tempfile.TemporaryDirectory(prefix="formant-benchmark-inputs-") as temporary:
        inputs = build_tracking_inputs(
            dataset,
            tracker,
            input_mode=tracking.input_mode,
            interval_type=tracking.interval_type,
            interval_padding_s=tracking.interval_padding_s,
            split=tracking.split,
            temporary_directory=Path(temporary),
        )
        condition = tracking_condition(
            tracker=tracker,
            inputs=inputs,
            tracker_config=tracker_config,
            dataset_fingerprint=fingerprint,
            input_mode=tracking.input_mode,
            dataset_tracker_config=dataset_tracker_config,
            experiment_config=experiment_layer,
            cli_parameters=cli_parameters,
            interval_type=tracking.interval_type,
            interval_padding_s=tracking.interval_padding_s,
            split=tracking.split,
        )
        run_root, run_action = _select_run(runs_root, experiment.name, condition.digest, resume=resume)
        if run_action == "reused":
            run = load_prediction_run(run_root)
        else:
            run = tracker.run(
                inputs,
                config=tracker_config,
                destination=run_root,
                dataset_fingerprint=fingerprint,
                dataset_name=dataset.manifest.name,
                dataset_tracker_config=dataset_tracker_config,
                experiment_config=experiment_layer,
                cli_parameters=cli_parameters,
                input_mode=tracking.input_mode,
                interval_type=tracking.interval_type,
                interval_padding_s=tracking.interval_padding_s,
                split=tracking.split,
                resume=run_action == "resumed",
                fail_fast=fail_fast,
                show_progress=show_progress,
            )

    evaluation, evaluation_action = _evaluate(
        experiment,
        dataset,
        run,
        fingerprint=fingerprint,
        destination=evaluations_root / experiment.name,
        overwrite=overwrite,
    )
    return ExperimentOutcome(run, run_action, evaluation, evaluation_action)


def _select_run(runs_root: Path, name: str, digest: str, *, resume: bool) -> tuple[Path, str]:
    """Choose where predictions come from: an existing matching run or ``runs/<name>``."""
    own = runs_root / name
    if own.exists():
        manifest = read_prediction_run_manifest(own)
        if manifest.configuration_digest != digest:
            raise ExperimentConflictError(
                f"{own} already exists from a different tracking condition. "
                f"Rename the experiment or remove {own}."
            )
        return own, _reuse_or_resume(own, manifest, resume=resume)

    matches = [
        (path, manifest)
        for path, manifest in _run_manifests(runs_root)
        if manifest.configuration_digest == digest
    ]
    for path, manifest in matches:
        if manifest.status == _COMPLETED:
            return path, "reused"
    if matches:
        path, manifest = matches[0]
        return path, _reuse_or_resume(path, manifest, resume=resume)
    return own, "tracked"


def _reuse_or_resume(path: Path, manifest: PredictionRunManifest, *, resume: bool) -> str:
    if manifest.status == _COMPLETED:
        return "reused"
    if resume:
        return "resumed"
    raise ExperimentConflictError(
        f"{path} has the same tracking condition but did not finish. Rerun with --resume to continue it."
    )


def _run_manifests(runs_root: Path) -> list[tuple[Path, PredictionRunManifest]]:
    if not runs_root.is_dir():
        return []
    manifests = []
    for path in sorted(runs_root.iterdir()):
        if not (path / "run_manifest.yaml").is_file():
            continue
        try:
            manifests.append((path, read_prediction_run_manifest(path)))
        except PredictionRunValidationError:
            continue  # An unreadable foreign directory can never match a digest.
    return manifests


def _evaluate(
    experiment: ExperimentConfig,
    dataset: PreparedDataset,
    run: PredictionRun,
    *,
    fingerprint: str,
    destination: Path,
    overwrite: bool,
) -> tuple[EvaluationResult, str]:
    settings = experiment.evaluation
    options = {
        "scope": settings.scope,
        "split": experiment.evaluation_split,
        "formants": settings.formants,
        "metrics": settings.metrics,
        "central_region": settings.central_region,
        "group_by": settings.group_by,
        "fdr_relative_threshold": settings.fdr_relative_threshold,
        "fdr_absolute_threshold_hz": settings.fdr_absolute_threshold_hz,
    }
    action = "evaluated"
    if destination.exists():
        digest = evaluation_settings_digest(
            dataset_fingerprint=fingerprint,
            prediction_run_digest=run.manifest.configuration_digest,
            **options,
        )
        try:
            existing_digest = read_evaluation_manifest(destination).settings_digest
        except EvaluationValidationError:
            existing_digest = None
        if existing_digest == digest:
            return load_evaluation_result(destination), "reused"
        if not overwrite:
            raise ExperimentConflictError(
                f"{destination} already exists with different evaluation settings or predictions. "
                "Use --overwrite to replace it."
            )
        action = "overwritten"

    result = evaluate(dataset, run, evaluation_id=experiment.name, **options)
    return write_evaluation_result(result, destination, overwrite=overwrite), action
