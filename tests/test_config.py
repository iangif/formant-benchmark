"""Tests for precedence and metadata-specific parameter resolution."""

import pytest

from formant_benchmark.config.resolution import check_override_rules, deep_merge, resolve_parameters
from formant_benchmark.exceptions import ConfigurationConflictError, ConfigurationError


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


_INPUTS = [
    {"gender": "female", "speaker_id": "s1"},
    {"gender": "male", "speaker_id": "s2", "age": None},
]


def test_override_rules_on_unknown_metadata_fields_fail() -> None:
    layer = {"overrides": [{"where": {"gendr": "male"}, "parameters": {"ceiling": 5000}}]}
    with pytest.raises(ConfigurationError) as error:
        check_override_rules(_INPUTS, {"the dataset's tracker_overrides": layer})
    message = str(error.value)
    assert "override 0 in the dataset's tracker_overrides uses metadata field(s) 'gendr'" in message
    assert "Available metadata fields: age, gender, speaker_id." in message


def test_override_rules_matching_no_input_warn() -> None:
    layers = {
        "the tracker config": {"overrides": [{"where": {"gender": "male"}, "parameters": {"ceiling": 5000}}]},
        "the experiment config": {
            "overrides": [
                {"where": {"gender": "Male"}, "parameters": {"ceiling": 5000}},
                {"where": {"age": 30}, "parameters": {"ceiling": 5000}},
            ]
        },
        "the dataset's tracker_overrides": None,
    }
    assert check_override_rules(_INPUTS, layers) == [
        "override 0 in the experiment config (gender='Male') matches no tracking input",
        "override 1 in the experiment config (age=30) matches no tracking input",
    ]
