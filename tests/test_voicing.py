"""Tests for explicit V1 voiced-feature failures."""

import pytest

from formant_benchmark.evaluation.scopes import require_implemented_scope
from formant_benchmark.exceptions import UnsupportedScopeError, UnsupportedVoicedFeatureError
from formant_benchmark.preparation.voicing import generate_voiced_intervals


def test_voiced_evaluation_scope_fails_explicitly() -> None:
    with pytest.raises(UnsupportedScopeError, match="Voiced scope is not implemented yet"):
        require_implemented_scope("voiced")


def test_benchmark_voicing_generation_fails_explicitly() -> None:
    with pytest.raises(UnsupportedVoicedFeatureError, match="Voiced scope is not implemented yet"):
        generate_voiced_intervals()


def test_supported_scopes_are_accepted() -> None:
    assert require_implemented_scope("all").value == "all"
    assert require_implemented_scope("vowels").value == "vowels"


def test_unknown_scope_is_domain_error() -> None:
    with pytest.raises(UnsupportedScopeError, match="Unsupported evaluation scope"):
        require_implemented_scope("segments")
