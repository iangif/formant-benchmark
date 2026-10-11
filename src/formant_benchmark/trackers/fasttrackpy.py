"""FastTrackPy 0.6.1 tracker adapter and parameter validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.trackers.base import TrackerAdapter, TrackerCapabilities
from formant_benchmark.trackers.parameters import nonnegative_float, positive_float, positive_int
from formant_benchmark.trackers.python_wrapper import python_wrapper_command, run_wrapper_check

FASTTRACKPY_VERSION = "0.6.1"
_LABEL = "FastTrackPy"

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
        return python_wrapper_command(
            config,
            module="formant_benchmark.tracker_wrappers.fasttrackpy",
            label=_LABEL,
            environment=f"fasttrackpy=={FASTTRACKPY_VERSION}",
        )

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Reject unknown or structurally unsafe FastTrackPy parameters."""
        unknown = sorted(set(parameters) - _ALLOWED_PARAMETERS)
        if unknown:
            raise ConfigurationError(
                "Unsupported FastTrackPy parameter(s): " + ", ".join(unknown)
            )

        minimum = positive_float(parameters, "min_max_formant", label=_LABEL)
        maximum = positive_float(parameters, "max_max_formant", label=_LABEL)
        if maximum <= minimum:
            raise ConfigurationError("FastTrackPy max_max_formant must be greater than min_max_formant.")
        positive_int(parameters, "nstep", label=_LABEL)
        n_formants = positive_int(parameters, "n_formants", label=_LABEL)
        if n_formants > 4:
            raise ConfigurationError(
                "FastTrackPy n_formants must be at most 4 because the benchmark canonical vocabulary is F1-F4."
            )
        positive_float(parameters, "window_length", label=_LABEL)
        positive_float(parameters, "time_step", label=_LABEL)
        nonnegative_float(parameters, "pre_emphasis_from", label=_LABEL)
        positive_float(parameters, "pitch_floor", label=_LABEL)
        positive_int(parameters, "smoother_order", label=_LABEL)

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
        return run_wrapper_check(
            self,
            config,
            label=_LABEL,
            details={"required_fasttrackpy_version": FASTTRACKPY_VERSION},
            version_key="fasttrackpy_version",
            expected_version=FASTTRACKPY_VERSION,
        )
