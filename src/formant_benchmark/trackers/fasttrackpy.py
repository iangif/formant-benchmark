"""FastTrackPy 0.6.1 tracker adapter and parameter validation."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.execution.backends import backend_from_config
from formant_benchmark.trackers.base import TrackerAdapter, TrackerCapabilities

FASTTRACKPY_VERSION = "0.6.1"

_ALLOWED_PARAMETERS = {
    "min_max_formant",
    "max_max_formant",
    "nstep",
    "n_formants",
    "window_length",
    "time_step",
    "pre_emphasis_from",
    "pitch_floor",
    "smoother_method",
    "smoother_order",
    "loss_method",
    "agg_method",
    "heuristics",
}
_ALLOWED_SMOOTHERS = {"dct_smooth_regression", "dct_smooth"}
_ALLOWED_LOSSES = {"lmse", "mse"}
_ALLOWED_AGGREGATORS = {"agg_sum"}
_ALLOWED_HEURISTICS = {
    "f1_max",
    "f4_min",
    "b2_max",
    "b3_max",
    "rhotic",
    "f3_f4_sep",
}


class FastTrackPyTracker(TrackerAdapter):
    """Run the official FastTrackPy 0.6.1 release through an isolated wrapper."""

    name = "fasttrackpy"
    version = FASTTRACKPY_VERSION
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
    default_configuration: ClassVar[Mapping[str, Any]] = {
        "execution": {
            "backend": "local",
            "timeout_s": 120,
            "checkpoint_every": 50,
        },
        "parameters": {
            "min_max_formant": 4000.0,
            "max_max_formant": 7000.0,
            "nstep": 20,
            "n_formants": 4,
            "window_length": 0.025,
            "time_step": 0.002,
            "pre_emphasis_from": 50.0,
            "pitch_floor": 75.0,
            "smoother_method": "dct_smooth_regression",
            "smoother_order": 5,
            "loss_method": "lmse",
            "agg_method": "agg_sum",
            "heuristics": [],
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
                "FastTrackPy requires execution.command pointing to the Python executable "
                "inside its isolated fasttrackpy==0.6.1 environment."
            )
        return (*prefix, "-m", "formant_benchmark.tracker_wrappers.fasttrackpy")

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Reject unknown or structurally unsafe FastTrackPy parameters."""
        unknown = sorted(set(parameters) - _ALLOWED_PARAMETERS)
        if unknown:
            raise ConfigurationError(
                "Unsupported FastTrackPy parameter(s): " + ", ".join(unknown)
            )

        minimum = _positive_float(parameters, "min_max_formant")
        maximum = _positive_float(parameters, "max_max_formant")
        if maximum <= minimum:
            raise ConfigurationError("FastTrackPy max_max_formant must be greater than min_max_formant.")
        _positive_int(parameters, "nstep")
        n_formants = _positive_int(parameters, "n_formants")
        if n_formants > 4:
            raise ConfigurationError(
                "FastTrackPy n_formants must be at most 4 because the benchmark canonical vocabulary is F1-F4."
            )
        _positive_float(parameters, "window_length")
        _positive_float(parameters, "time_step")
        _nonnegative_float(parameters, "pre_emphasis_from")
        _positive_float(parameters, "pitch_floor")
        _positive_int(parameters, "smoother_order")

        if parameters.get("smoother_method") not in _ALLOWED_SMOOTHERS:
            raise ConfigurationError(
                "FastTrackPy smoother_method must be one of: " + ", ".join(sorted(_ALLOWED_SMOOTHERS))
            )
        if parameters.get("loss_method") not in _ALLOWED_LOSSES:
            raise ConfigurationError(
                "FastTrackPy loss_method must be one of: " + ", ".join(sorted(_ALLOWED_LOSSES))
            )
        if parameters.get("agg_method") not in _ALLOWED_AGGREGATORS:
            raise ConfigurationError("FastTrackPy agg_method must be 'agg_sum'.")

        heuristics = parameters.get("heuristics")
        if not isinstance(heuristics, list) or any(value not in _ALLOWED_HEURISTICS for value in heuristics):
            raise ConfigurationError(
                "FastTrackPy heuristics must be a list containing only: "
                + ", ".join(sorted(_ALLOWED_HEURISTICS))
            )
        if len(set(heuristics)) != len(heuristics):
            raise ConfigurationError("FastTrackPy heuristics must not contain duplicates.")

    def check_environment(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Check the runtime and confirm the isolated environment has FastTrackPy 0.6.1."""
        backend = backend_from_config(config)
        command = tuple(self.wrapper_command(config))
        result = backend.check(command)
        result["tracker"] = self.name
        result["required_fasttrackpy_version"] = FASTTRACKPY_VERSION
        if not result.get("available"):
            return result

        execution = config.get("execution", {})
        if not isinstance(execution, Mapping) or execution.get("backend", "local") != "local":
            result["package_check"] = "not_run_for_container_backend"
            return result

        environment = os.environ.copy()
        extra_environment = execution.get("environment", {})
        if isinstance(extra_environment, Mapping):
            environment.update({str(key): str(value) for key, value in extra_environment.items()})
        working_directory = execution.get("working_directory")
        try:
            completed = subprocess.run(
                [*command, "--check"],
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["available"] = False
            result.setdefault("problems", []).append(f"FastTrackPy wrapper check failed: {exc}")
            return result

        try:
            details = json.loads(completed.stdout.strip()) if completed.stdout.strip() else {}
        except json.JSONDecodeError:
            details = {}
        result["package_check"] = details
        observed_version = details.get("fasttrackpy_version")
        if observed_version is not None and observed_version != FASTTRACKPY_VERSION:
            result["available"] = False
            result.setdefault("problems", []).append(
                "FastTrackPy version mismatch: expected "
                f"{FASTTRACKPY_VERSION}, found {observed_version}."
            )
        elif completed.returncode != 0:
            result["available"] = False
            message = (
                details.get("error")
                or completed.stderr.strip()
                or completed.stdout.strip()
                or "unknown wrapper check failure"
            )
            result.setdefault("problems", []).append(f"FastTrackPy wrapper check failed: {message}")
        return result


def _positive_float(parameters: Mapping[str, Any], key: str) -> float:
    try:
        value = float(parameters[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"FastTrackPy {key} must be a positive number.") from exc
    if value <= 0:
        raise ConfigurationError(f"FastTrackPy {key} must be a positive number.")
    return value


def _nonnegative_float(parameters: Mapping[str, Any], key: str) -> float:
    try:
        value = float(parameters[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"FastTrackPy {key} must be a non-negative number.") from exc
    if value < 0:
        raise ConfigurationError(f"FastTrackPy {key} must be a non-negative number.")
    return value


def _positive_int(parameters: Mapping[str, Any], key: str) -> int:
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"FastTrackPy {key} must be a positive integer.")
    return value
