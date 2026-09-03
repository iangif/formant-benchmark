"""Minimal tracker capability types and registry used by later integrations."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.registry import Registry


@dataclass(frozen=True, slots=True)
class TrackerCapabilities:
    """Static capabilities exposed before expensive tracker execution."""

    formants: frozenset[Formant]
    input_modes: frozenset[TrackingInputMode]
    interval_types: frozenset[str] = frozenset()
    requires_audio: bool = True

    def __post_init__(self) -> None:
        if not self.formants:
            raise ValueError("A tracker must support at least one canonical formant.")


@dataclass(frozen=True, slots=True)
class TrackingInput:
    """One normalized invocation presented to a tracker-side wrapper."""

    item_id: str
    input_unit_id: str
    audio_path: Path
    duration_s: float
    source_start_s: float
    source_end_s: float
    metadata: Mapping[str, Any]
    intervals: tuple[Mapping[str, Any], ...] = ()


class TrackerAdapter(ABC):
    """Batch-oriented tracker contract implemented through an isolated wrapper."""

    name: str
    version: str = "1"
    capabilities: TrackerCapabilities
    default_configuration: Mapping[str, Any] = {}

    def wrapper_command(self, config: Mapping[str, Any]) -> Sequence[str]:
        """Return the tracker-environment command, excluding protocol arguments."""
        execution = config.get("execution", {})
        command = execution.get("command") if isinstance(execution, Mapping) else None
        if isinstance(command, str):
            return (command,)
        if isinstance(command, Sequence) and not isinstance(command, (str, bytes)) and command:
            return tuple(str(part) for part in command)
        raise ValueError(f"Tracker '{self.name}' requires execution.command.")

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Validate tracker-owned parameters before execution."""

    def run(
        self,
        inputs: Sequence[TrackingInput],
        *,
        config: Mapping[str, Any],
        destination: str | Path,
        dataset_fingerprint: str,
        dataset_name: str,
        dataset_tracker_config: Mapping[str, Any] | None = None,
        experiment_config: Mapping[str, Any] | None = None,
        cli_parameters: Mapping[str, Any] | None = None,
        input_mode: TrackingInputMode,
        interval_type: str | None = None,
        split: str | None = None,
        resume: bool = False,
        fail_fast: bool = False,
        show_progress: bool = False,
    ) -> Any:
        """Run normalized inputs using the generic Phase 5 orchestration."""
        from formant_benchmark.runs.runner import execute_prediction_run

        return execute_prediction_run(
            tracker=self,
            inputs=inputs,
            tracker_config=config,
            dataset_tracker_config=dataset_tracker_config,
            experiment_config=experiment_config,
            cli_parameters=cli_parameters,
            destination=destination,
            dataset_fingerprint=dataset_fingerprint,
            dataset_name=dataset_name,
            input_mode=input_mode,
            interval_type=interval_type,
            split=split,
            resume=resume,
            fail_fast=fail_fast,
            show_progress=show_progress,
        )


TRACKER_REGISTRY: Registry[type[TrackerAdapter]] = Registry()
