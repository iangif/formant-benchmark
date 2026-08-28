"""Evaluation-scope validation, including explicit V1 voiced failure."""

from __future__ import annotations

from formant_benchmark.data.models import EvaluationScope
from formant_benchmark.exceptions import UnsupportedScopeError
from formant_benchmark.preparation.voicing import VOICED_NOT_IMPLEMENTED_MESSAGE


def require_implemented_scope(scope: str | EvaluationScope) -> EvaluationScope:
    """Return a supported scope and fail early for unavailable/unknown scopes."""
    try:
        parsed = EvaluationScope(scope)
    except ValueError as exc:
        raise UnsupportedScopeError(
            f"Unsupported evaluation scope '{scope}'. Supported evaluation scopes: all, vowels."
        ) from exc
    if parsed is EvaluationScope.VOICED:
        raise UnsupportedScopeError(VOICED_NOT_IMPLEMENTED_MESSAGE)
    return parsed
