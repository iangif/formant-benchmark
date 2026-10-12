"""Experiment YAML schema: one dataset, one tracker, one tracking and one evaluation condition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from formant_benchmark.config.loading import load_yaml
from formant_benchmark.data.models import EvaluationScope, Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError


class TrackingSettings(BaseModel):
    """How the tracker is run; every field here can change the predictions."""

    model_config = ConfigDict(extra="forbid")

    input_mode: TrackingInputMode
    interval_type: str | None = None
    interval_padding_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    split: str | None = None
    # Experiment-level tracker parameter layer (precedence 4 of 5).
    parameters: dict[str, Any] = Field(default_factory=dict)
    overrides: list[dict[str, Any]] = Field(default_factory=list)

    def parameter_layer(self) -> dict[str, Any]:
        """Return the layer passed to parameter resolution and the run digest."""
        layer: dict[str, Any] = {}
        if self.parameters:
            layer["parameters"] = dict(self.parameters)
        if self.overrides:
            layer["overrides"] = [dict(rule) for rule in self.overrides]
        return layer


class EvaluationSettings(BaseModel):
    """How the persisted predictions are scored; changing these never retracks."""

    model_config = ConfigDict(extra="forbid")

    scope: EvaluationScope
    # None means "the tracking split".
    split: str | None = None
    formants: list[Formant] | None = None
    metrics: list[str] | None = None
    central_region: float | None = None
    fdr_relative_threshold: float = 0.30
    fdr_absolute_threshold_hz: float = 300.0


class ExperimentConfig(BaseModel):
    """One experimental condition, run by ``formant-benchmark experiment run``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    tracker: str = Field(min_length=1)
    tracker_config: str | None = None
    tracking: TrackingSettings
    evaluation: EvaluationSettings

    @field_validator("name")
    @classmethod
    def _name_is_directory_safe(cls, value: str) -> str:
        if Path(value).name != value or value in {".", ".."}:
            raise ValueError("must be a plain name; it becomes runs/<name> and evaluations/<name>")
        return value

    @property
    def evaluation_split(self) -> str | None:
        return self.evaluation.split if self.evaluation.split is not None else self.tracking.split


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment file; ``name`` defaults to the file name."""
    config_path = Path(path)
    raw = load_yaml(config_path)
    raw.setdefault("name", config_path.stem)
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "\n".join(
            f"  {'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(f"Invalid experiment config '{config_path}':\n{problems}") from exc

