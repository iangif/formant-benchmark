"""Tests for precedence and metadata-specific parameter resolution."""

import pytest

from formant_benchmark.config.resolution import deep_merge, resolve_parameters
from formant_benchmark.exceptions import ConfigurationConflictError


def test_deep_merge_honors_later_precedence() -> None:
    resolved = deep_merge(
        {"tracker": {"parameters": {"a": 1, "b": 2}}},
        {"tracker": {"parameters": {"b": 3}}},
    )
    assert resolved == {"tracker": {"parameters": {"a": 1, "b": 3}}}


def test_parameter_rules_apply_least_to_most_specific_then_layer_precedence() -> None:
    built_in = {"parameters": {"ceiling": 5000, "step": 10}}
    dataset = {
        "overrides": [
            {"where": {"gender": "female"}, "parameters": {"ceiling": 5500}},
            {"where": {"gender": "female", "source": "vtr"}, "parameters": {"ceiling": 6000}},
        ]
    }
    experiment = {"parameters": {"step": 5}}
    resolved = resolve_parameters({"gender": "female", "source": "vtr"}, built_in, dataset, experiment)
    assert resolved == {"ceiling": 6000, "step": 5}


def test_equally_specific_conflicting_rules_fail() -> None:
    layer = {
        "overrides": [
            {"where": {"gender": "female"}, "parameters": {"ceiling": 5500}},
            {"where": {"source": "vtr"}, "parameters": {"ceiling": 6000}},
        ]
    }
    with pytest.raises(ConfigurationConflictError):
        resolve_parameters({"gender": "female", "source": "vtr"}, layer)


def test_yaml_loading(tmp_path) -> None:
    from formant_benchmark.config.loading import load_yaml

    path = tmp_path / "config.yaml"
    path.write_text("dataset:\n  adapter: synthetic\n", encoding="utf-8")
    assert load_yaml(path) == {"dataset": {"adapter": "synthetic"}}
