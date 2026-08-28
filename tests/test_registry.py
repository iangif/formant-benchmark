"""Tests for additive extension registries."""

import pytest

from formant_benchmark.exceptions import DuplicateRegistrationError, UnknownRegistrationError
from formant_benchmark.registry import Registry


def test_registry_is_deterministic_and_rejects_replacement() -> None:
    registry: Registry[int] = Registry()
    registry.register("zeta", 2)
    registry.register("alpha", 1)
    assert registry.names() == ("alpha", "zeta")
    assert registry.get("alpha") == 1
    with pytest.raises(DuplicateRegistrationError):
        registry.register("alpha", 3)
    with pytest.raises(UnknownRegistrationError):
        registry.get("missing")
