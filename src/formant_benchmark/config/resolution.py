"""Configuration precedence and metadata-specific tracker parameter resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from formant_benchmark.exceptions import ConfigurationConflictError, ConfigurationError


def deep_merge(*layers: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively merge mappings from lowest to highest precedence."""
    merged: dict[str, Any] = {}
    for layer in layers:
        if not layer:
            continue
        merged = _merge_two(merged, layer)
    return merged


def _merge_two(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_two(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_parameters(
    metadata: Mapping[str, Any],
    *layers: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve tracker parameters across precedence layers and metadata rules.

    Each layer may contain ``parameters`` and ``overrides``. Matching overrides are
    applied least-specific to most-specific. Within one layer, equally specific
    rules that assign different values to the same parameter are ambiguous and fail.
    Later layers always have higher precedence than earlier layers.
    """
    resolved: dict[str, Any] = {}
    for layer_index, layer in enumerate(layers):
        if not layer:
            continue
        parameters = layer.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ConfigurationError(f"Layer {layer_index} 'parameters' must be a mapping.")
        resolved = deep_merge(resolved, parameters)
    
        overrides = layer.get("overrides", [])
        if not isinstance(overrides, Sequence) or isinstance(overrides, (str, bytes)):
            raise ConfigurationError(f"Layer {layer_index} 'overrides' must be a list.")
    
        matching: list[tuple[int, int, Mapping[str, Any]]] = []
        for rule_index, rule in enumerate(overrides):
            if not isinstance(rule, Mapping):
                raise ConfigurationError(f"Override {rule_index} in layer {layer_index} must be a mapping.")
            where = rule.get("where", {})
            rule_parameters = rule.get("parameters", {})
            if not isinstance(where, Mapping) or not isinstance(rule_parameters, Mapping):
                raise ConfigurationError("Override 'where' and 'parameters' must be mappings.")
            if all(metadata.get(key) == value for key, value in where.items()):
                matching.append((len(where), rule_index, rule_parameters))
    

        _reject_equal_specificity_conflicts(matching, layer_index)
        for _, _, rule_parameters in sorted(matching, key=lambda value: (value[0], value[1])):
            resolved = deep_merge(resolved, rule_parameters)
    
    return resolved


def _reject_equal_specificity_conflicts(
    matching: list[tuple[int, int, Mapping[str, Any]]], layer_index: int
) -> None:
    by_specificity: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
    for specificity, rule_index, parameters in matching:
        by_specificity.setdefault(specificity, []).append((rule_index, parameters))

    for specificity, rules in by_specificity.items():
        seen: dict[str, tuple[Any, int]] = {}
        for rule_index, parameters in rules:
            for key, value in parameters.items():
                if key in seen and seen[key][0] != value:
                    previous_index = seen[key][1]
                    raise ConfigurationConflictError(
                        "Conflicting equally specific metadata overrides in "
                        f"layer {layer_index}: rules {previous_index} and {rule_index} "
                        f"both set '{key}' at specificity {specificity}."
                    )
                seen[key] = (value, rule_index)