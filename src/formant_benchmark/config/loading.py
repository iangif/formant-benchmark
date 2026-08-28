"""YAML config loading with predictable top-level mapping semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from formant_benchmark.exceptions import ConfigurationError

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, treating an empty file as an empty config."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not load configuration '{config_path}': {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration '{config_path}' must contain a top-level mapping.")
    
    return raw

