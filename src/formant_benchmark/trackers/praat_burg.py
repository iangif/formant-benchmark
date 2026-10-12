"""Praat Burg (via Parselmouth 0.4.7) tracker adapter and parameter validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.trackers.base import TrackerAdapter, TrackerCapabilities
from formant_benchmark.trackers.parameters import positive_float
from formant_benchmark.trackers.python_wrapper import python_wrapper_command, run_wrapper_check

PARSELMOUTH_VERSION = "0.4.7"
# Parselmouth bundles a fixed Praat build; this is the one shipped with 0.4.7.
PRAAT_VERSION = "6.1.38"
_LABEL = "Praat Burg"

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
        return python_wrapper_command(
            config,
            module="formant_benchmark.tracker_wrappers.praat_burg",
            label=_LABEL,
            environment=f"praat-parselmouth=={PARSELMOUTH_VERSION}",
        )

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Reject unknown or structurally unsafe Burg parameters."""
        unknown = sorted(set(parameters) - _ALLOWED_PARAMETERS)
        if unknown:
            raise ConfigurationError("Unsupported Praat Burg parameter(s): " + ", ".join(unknown))

        if parameters.get("time_step") is not None:
            positive_float(parameters, "time_step", label=_LABEL)
        # Burg fits 2 * max_number_of_formants LPC poles, so Praat rounds this to the
        # nearest half; requiring a half-integer avoids silently analysing another value.
        max_formants = positive_float(parameters, "max_number_of_formants", label=_LABEL)
        if (max_formants * 2) != int(max_formants * 2):
            raise ConfigurationError("Praat Burg max_number_of_formants must be a multiple of 0.5.")
        positive_float(parameters, "maximum_formant", label=_LABEL)
        positive_float(parameters, "window_length", label=_LABEL)
        positive_float(parameters, "pre_emphasis_from", label=_LABEL)

    def check_environment(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Check the runtime and confirm the isolated environment has Parselmouth 0.4.7."""
        return run_wrapper_check(
            self,
            config,
            label="Parselmouth",
            details={
                "required_parselmouth_version": PARSELMOUTH_VERSION,
                "expected_praat_version": PRAAT_VERSION,
            },
            version_key="parselmouth_version",
            expected_version=PARSELMOUTH_VERSION,
        )
