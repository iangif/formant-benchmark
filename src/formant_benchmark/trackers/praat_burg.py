"""Praat Burg (via Parselmouth 0.4.7) tracker adapter and parameter validation."""

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

PARSELMOUTH_VERSION = "0.4.7"
# Parselmouth bundles a fixed Praat build; this is the one shipped with 0.4.7.
PRAAT_VERSION = "6.1.38"

_ALLOWED_PARAMETERS = {
    "time_step",
    "max_number_of_formants",
    "maximum_formant",
    "window_length",
    "pre_emphasis_from",
}


class PraatBurgTracker(TrackerAdapter):
    """Run Praat's "Sound: To Formant (burg)..." through an isolated Parselmouth wrapper."""

    name = "praat_burg"
    version = PARSELMOUTH_VERSION
    capabilities = TrackerCapabilities(
        formants=frozenset(Formant),
        input_modes=frozenset(
            {
                TrackingInputMode.FULL_ITEM,
                TrackingInputMode.CROPPED_INTERVALS,
            }
        ),
        interval_types=frozenset({"phone", "vowel", "word"}),
        requires_audio=True,
    )
    # Praat's own dialog defaults. A null time_step lets Praat choose 25% of the
    # window length, exactly as the Praat GUI's "0.0 (= auto)" does.
    default_configuration: ClassVar[Mapping[str, Any]] = {
        "execution": {
            "backend": "local",
            "timeout_s": 120,
            "checkpoint_every": 50,
        },
        "parameters": {
            "time_step": None,
            "max_number_of_formants": 5.0,
            "maximum_formant": 5500.0,
            "window_length": 0.025,
            "pre_emphasis_from": 50.0,
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
                "Praat Burg requires execution.command pointing to the Python executable "
                f"inside its isolated praat-parselmouth=={PARSELMOUTH_VERSION} environment."
            )
        return (*prefix, "-m", "formant_benchmark.tracker_wrappers.praat_burg")

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Reject unknown or structurally unsafe Burg parameters."""
        unknown = sorted(set(parameters) - _ALLOWED_PARAMETERS)
        if unknown:
            raise ConfigurationError("Unsupported Praat Burg parameter(s): " + ", ".join(unknown))

        if parameters.get("time_step") is not None:
            _positive_float(parameters, "time_step")
        # Burg fits 2 * max_number_of_formants LPC poles, so Praat rounds this to the
        # nearest half; requiring a half-integer avoids silently analysing another value.
        max_formants = _positive_float(parameters, "max_number_of_formants")
        if (max_formants * 2) != int(max_formants * 2):
            raise ConfigurationError("Praat Burg max_number_of_formants must be a multiple of 0.5.")
        _positive_float(parameters, "maximum_formant")
        _positive_float(parameters, "window_length")
        _positive_float(parameters, "pre_emphasis_from")

    def check_environment(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Check the runtime and confirm the isolated environment has Parselmouth 0.4.7."""
        backend = backend_from_config(config)
        command = tuple(self.wrapper_command(config))
        result = backend.check(command)
        result["tracker"] = self.name
        result["required_parselmouth_version"] = PARSELMOUTH_VERSION
        result["expected_praat_version"] = PRAAT_VERSION
        if not result.get("available"):
            return result

        if not isinstance(backend, LocalExecutionBackend):
            result["package_check"] = "not_run_for_container_backend"
            return result

        resolved_command = backend.resolve_command(command)
        try:
            completed = subprocess.run(
                [*resolved_command, "--check"],
                cwd=backend.working_directory,
                env=backend.process_environment(),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["available"] = False
            result.setdefault("problems", []).append(f"Praat Burg wrapper check failed: {exc}")
            return result

        try:
            details = json.loads(completed.stdout.strip()) if completed.stdout.strip() else {}
        except json.JSONDecodeError:
            details = {}
        result["package_check"] = details
        observed_version = details.get("parselmouth_version")
        if observed_version is not None and observed_version != PARSELMOUTH_VERSION:
            result["available"] = False
            result.setdefault("problems", []).append(
                "Parselmouth version mismatch: expected "
                f"{PARSELMOUTH_VERSION}, found {observed_version}."
            )
        elif completed.returncode != 0:
            result["available"] = False
            message = (
                details.get("error")
                or completed.stderr.strip()
                or completed.stdout.strip()
                or "unknown wrapper check failure"
            )
            result.setdefault("problems", []).append(f"Praat Burg wrapper check failed: {message}")
        return result


def _positive_float(parameters: Mapping[str, Any], key: str) -> float:
    value = parameters.get(key)
    if isinstance(value, bool):
        raise ConfigurationError(f"Praat Burg {key} must be a positive number.")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Praat Burg {key} must be a positive number.") from exc
    if not numeric > 0:
        raise ConfigurationError(f"Praat Burg {key} must be a positive number.")
    return numeric
