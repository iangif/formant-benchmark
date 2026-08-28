"""Reserved voiced-segment hooks; V1 intentionally provides no voicing detector."""

from __future__ import annotations

from typing import NoReturn

from formant_benchmark.exceptions import UnsupportedVoicedFeatureError

VOICED_NOT_IMPLEMENTED_MESSAGE = (
    "Voiced scope is not implemented yet. Supported evaluation scopes: all, vowels."
)


def require_voiced_feature() -> NoReturn:
    """Fail explicitly whenever a V1 operation requires voiced-segment functionality."""
    raise UnsupportedVoicedFeatureError(VOICED_NOT_IMPLEMENTED_MESSAGE)


def generate_voiced_intervals(*_args: object, **_kwargs: object) -> NoReturn:
    """Reserved future entry point for benchmark-generated voiced intervals."""
    require_voiced_feature()
