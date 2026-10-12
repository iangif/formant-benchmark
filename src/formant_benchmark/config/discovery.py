"""Conventional repository locations and name-or-path lookup for CLI inputs.

Names resolve relative to the current working directory, matching how explicit
output paths are interpreted: ``vtr`` means ``prepared/vtr`` for a dataset and
``configs/datasets/vtr.yaml`` for a dataset config. An existing path always wins.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from formant_benchmark.config.loading import load_yaml
from formant_benchmark.exceptions import ConfigurationError

CONFIGS_ROOT = Path("configs")
PREPARED_ROOT = Path("prepared")
RUNS_ROOT = Path("runs")
EVALUATIONS_ROOT = Path("evaluations")


def resolve_config_file(value: str | Path, kind: str) -> Path:
    """Return an existing config file path, or ``configs/<kind>/<value>.yaml``.

    *kind* is a config directory name such as ``datasets``, ``trackers``, or
    ``experiments``.
    """
    path = Path(value)
    if path.is_file():
        return path
    candidate = CONFIGS_ROOT / kind / f"{value}.yaml"
    if _is_bare_name(value) and candidate.is_file():
        return candidate
    raise ConfigurationError(
        f"Could not find config '{value}': it is not a file and {candidate} does not exist."
    )


def resolve_prepared_dataset(value: str | Path) -> Path:
    """Return an existing prepared-dataset directory, or ``prepared/<value>``."""
    path = Path(value)
    if path.is_dir():
        return path
    candidate = PREPARED_ROOT / str(value)
    if _is_bare_name(value) and candidate.is_dir():
        return candidate
    raise ConfigurationError(
        f"Could not find prepared dataset '{value}': it is not a directory and {candidate} does not exist."
    )


def load_tracker_config(value: str | Path | None, tracker_name: str) -> dict[str, Any]:
    """Load an explicit tracker config, else ``configs/trackers/<tracker>.yaml`` when present.

    A missing default file means an empty config, so trackers that need no
    configuration keep working. A config naming a different tracker is an error.
    """
    if value is not None:
        path: Path | None = resolve_config_file(value, "trackers")
    else:
        default = CONFIGS_ROOT / "trackers" / f"{tracker_name}.yaml"
        path = default if default.is_file() else None
    config = load_yaml(path) if path is not None else {}
    configured_name = config.get("tracker")
    if configured_name is not None and configured_name != tracker_name:
        raise ConfigurationError(
            f"Tracker config {path} declares '{configured_name}' but '{tracker_name}' was selected."
        )
    return config


def timestamped_name(*parts: str) -> str:
    """Return a readable default output name such as ``20261011-200501-vtr-praat_burg``."""
    return "-".join([datetime.now().strftime("%Y%m%d-%H%M%S"), *parts])


def _is_bare_name(value: str | Path) -> bool:
    text = str(value)
    return bool(text) and Path(text).name == text and text not in {".", ".."}
