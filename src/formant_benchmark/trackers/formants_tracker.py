"""FormantsTracker adapter, capabilities, validation, and environment checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.trackers.base import TrackerAdapter, TrackerCapabilities
from formant_benchmark.trackers.python_wrapper import python_wrapper_command, run_wrapper_check

FORMANTS_TRACKER_MODEL_SAMPLE_RATE = 16000
FORMANTS_TRACKER_DEFAULT_CHECKPOINT = "saved_ckpts/196_1125.ckpt"
FORMANTS_TRACKER_TORCH_VERSION = "1.13.1"
FORMANTS_TRACKER_TORCHAUDIO_VERSION = "0.13.1"

_LABEL = "FormantsTracker"
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
        return python_wrapper_command(
            config,
            module="formant_benchmark.tracker_wrappers.formants_tracker",
            label=_LABEL,
            environment="FormantsTracker",
        )

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
        return run_wrapper_check(
            self,
            config,
            label=_LABEL,
            details={"model_sample_rate_hz": FORMANTS_TRACKER_MODEL_SAMPLE_RATE},
            check_arguments=self._check_arguments,
        )

    def _check_arguments(self, config: Mapping[str, Any]) -> Sequence[str]:
        # The wrapper's --check loads the configured checkpoint on the configured
        # device, so those parameters must be valid before the check can run.
        parameters = config.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ConfigurationError("FormantsTracker parameters must be a mapping.")
        self.validate_parameters(parameters)
        return (
            "--checkpoint",
            str(parameters["checkpoint"]),
            "--device",
            str(parameters["device"]),
        )
