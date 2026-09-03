"""Deterministic test-only tracker used to verify generic machinery."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.trackers.base import TrackerAdapter, TrackerCapabilities


class SyntheticTracker(TrackerAdapter):
    """Predict deterministic analytic tracks; never use for scientific results."""

    name = "synthetic"
    version = "1"
    capabilities = TrackerCapabilities(
        formants=frozenset(Formant),
        input_modes=frozenset(TrackingInputMode),
        interval_types=frozenset({"phone", "vowel", "word"}),
        requires_audio=False,
    )
    default_configuration: ClassVar[Mapping[str, Any]] = {
        "execution": {"backend": "local", "timeout_s": 30, "checkpoint_every": 50},
        "parameters": {
            "frame_step_s": 0.01,
            "base_f1": 500.0,
            "base_f2": 1500.0,
            "base_f3": 2500.0,
            "base_f4": 3500.0,
            "slope_hz_per_s": 10.0,
            "offset_hz": 0.0,
            "omit_formants": [],
        },
    }

    def wrapper_command(self, config: Mapping[str, Any]) -> Sequence[str]:
        execution = config.get("execution", {})
        if isinstance(execution, Mapping) and execution.get("command"):
            return super().wrapper_command(config)
        return (sys.executable, "-m", "formant_benchmark.tracker_wrappers.synthetic")

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        step = float(parameters.get("frame_step_s", 0))
        if step <= 0:
            raise ConfigurationError("Synthetic tracker frame_step_s must be greater than zero.")
        omit = parameters.get("omit_formants", [])
        if not isinstance(omit, list) or any(value not in {member.value for member in Formant} for value in omit):
            raise ConfigurationError("Synthetic tracker omit_formants must be a list containing only F1-F4.")
        if len(omit) == len(Formant):
            raise ConfigurationError("Synthetic tracker cannot omit every formant.")
