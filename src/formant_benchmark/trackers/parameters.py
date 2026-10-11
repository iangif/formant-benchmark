"""Small numeric validators shared by tracker-owned parameter vocabularies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from formant_benchmark.exceptions import ConfigurationError


def positive_float(parameters: Mapping[str, Any], key: str, *, label: str) -> float:
    """Return ``parameters[key]`` as a finite number greater than zero."""
    value = _finite_float(parameters, key, label=label, requirement="a positive number")
    if value <= 0:
        raise ConfigurationError(f"{label} {key} must be a positive number.")
    return value


def nonnegative_float(parameters: Mapping[str, Any], key: str, *, label: str) -> float:
    """Return ``parameters[key]`` as a finite number greater than or equal to zero."""
    value = _finite_float(parameters, key, label=label, requirement="a non-negative number")
    if value < 0:
        raise ConfigurationError(f"{label} {key} must be a non-negative number.")
    return value


def positive_int(parameters: Mapping[str, Any], key: str, *, label: str) -> int:
    """Return ``parameters[key]`` as an integer greater than zero; floats are rejected."""
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{label} {key} must be a positive integer.")
    return value


def _finite_float(parameters: Mapping[str, Any], key: str, *, label: str, requirement: str) -> float:
    # bool is an int subclass, so YAML `true` would otherwise silently become 1.0.
    value = parameters.get(key)
    message = f"{label} {key} must be {requirement}."
    if isinstance(value, bool):
        raise ConfigurationError(message)
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(message) from exc
    if not math.isfinite(numeric):
        raise ConfigurationError(message)
    return numeric
