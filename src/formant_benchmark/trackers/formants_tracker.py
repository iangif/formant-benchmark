"""FormantsTracker adapter, capabilities, validation, and environment checks."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.execution.backends import (
    LocalExecutionBackend,
    backend_from_config,
)
from formant_benchmark.trackers.base import TrackerAdapter, TrackerCapabilities

FORMANTS_TRACKER_MODEL_SAMPLE_RATE = 16000
FORMANTS_TRACKER_DEFAULT_CHECKPOINT = "saved_ckpts/196_1125.ckpt"
FORMANTS_TRACKER_TORCH_VERSION = "1.13.1"
FORMANTS_TRACKER_TORCHAUDIO_VERSION = "0.13.1"

_ALLOWED_PARAMETERS = {"checkpoint", "device"}
_ALLOWED_DEVICES = {"auto", "cpu", "cuda"}


class FormantsTrackerTracker(TrackerAdapter):
    """Run the MLSpeech FormantsTracker release through an isolated wrapper."""

    name = "formants_tracker"
    version = "1"
    capabilities = TrackerCapabilities(
        formants=frozenset({Formant.F1, Formant.F2, Formant.F3}),
        input_modes=frozenset(
            {
                TrackingInputMode.FULL_ITEM,
                TrackingInputMode.CROPPED_INTERVALS,
            }
        ),
        interval_types=frozenset({"phone", "vowel", "word"}),
        requires_audio=True,
    )
    default_configuration: ClassVar[Mapping[str, Any]] = {
        "execution": {
            "backend": "local",
            "timeout_s": 120,
            "checkpoint_every": 50,
        },
        "parameters": {
            "checkpoint": FORMANTS_TRACKER_DEFAULT_CHECKPOINT,
            "device": "auto",
        },
    }

    def wrapper_command(self, config: Mapping[str, Any]) -> Sequence[str]:
        """Return the configured tracker Python plus the benchmark wrapper module."""
        execution = config.get("execution", {})
        command = execution.get("command") if isinstance(execution, Mapping) else None
        if isinstance(command, str) and command.strip():
            prefix = (command,)
        elif isinstance(command, Sequence) and not isinstance(command, (str, bytes)) and command:
            prefix = tuple(str(part) for part in command)
        else:
            raise ConfigurationError(
                "FormantsTracker requires execution.command pointing to the Python executable "
                "inside its isolated FormantsTracker environment."
            )
        return (*prefix, "-m", "formant_benchmark.tracker_wrappers.formants_tracker")

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Validate the small set of inference choices exposed by this integration."""
        unknown = sorted(set(parameters) - _ALLOWED_PARAMETERS)
        if unknown:
            raise ConfigurationError(
                "Unsupported FormantsTracker parameter(s): " + ", ".join(unknown)
            )

        checkpoint = parameters.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ConfigurationError("FormantsTracker checkpoint must be a non-empty path string.")

        device = parameters.get("device")
        if device not in _ALLOWED_DEVICES:
            raise ConfigurationError(
                "FormantsTracker device must be one of: " + ", ".join(sorted(_ALLOWED_DEVICES))
            )

    def check_environment(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Check the isolated runtime, upstream source tree, checkpoint, and pinned Torch stack."""
        backend = backend_from_config(config)
        command = tuple(self.wrapper_command(config))
        result = backend.check(command)
        result["tracker"] = self.name
        result["model_sample_rate_hz"] = FORMANTS_TRACKER_MODEL_SAMPLE_RATE
        if not result.get("available"):
            return result

        if not isinstance(backend, LocalExecutionBackend):
            result["package_check"] = "not_run_for_container_backend"
            return result

        parameters = config.get("parameters", {})
        if not isinstance(parameters, Mapping):
            result["available"] = False
            result.setdefault("problems", []).append("FormantsTracker parameters must be a mapping.")
            return result
        try:
            self.validate_parameters(parameters)
        except ConfigurationError as exc:
            result["available"] = False
            result.setdefault("problems", []).append(str(exc))
            return result

        resolved_command = backend.resolve_command(command)
        try:
            completed = subprocess.run(
                [
                    *resolved_command,
                    "--check",
                    "--checkpoint",
                    str(parameters["checkpoint"]),
                    "--device",
                    str(parameters["device"]),
                ],
                cwd=backend.working_directory,
                env=backend.process_environment(),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["available"] = False
            result.setdefault("problems", []).append(f"FormantsTracker wrapper check failed: {exc}")
            return result

        try:
            details = json.loads(completed.stdout.strip()) if completed.stdout.strip() else {}
        except json.JSONDecodeError:
            details = {}
        result["package_check"] = details
        if completed.returncode != 0:
            result["available"] = False
            message = (
                details.get("error")
                or completed.stderr.strip()
                or completed.stdout.strip()
                or "unknown wrapper check failure"
            )
            result.setdefault("problems", []).append(
                f"FormantsTracker wrapper check failed: {message}"
            )
        return result
