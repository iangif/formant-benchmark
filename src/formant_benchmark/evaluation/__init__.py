"""Evaluation engine for normalized PredictionRuns and prepared gold datasets."""

from formant_benchmark.evaluation.evaluator import evaluate
from formant_benchmark.evaluation.io import (
    inspect_evaluation_result,
    load_evaluation_result,
    read_evaluation_manifest,
    write_evaluation_result,
)

__all__ = [
    "evaluate",
    "inspect_evaluation_result",
    "load_evaluation_result",
    "read_evaluation_manifest",
    "write_evaluation_result",
]
