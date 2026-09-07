"""Built-in V1 evaluation metrics."""

from formant_benchmark.evaluation.metrics.avg import avg
from formant_benchmark.evaluation.metrics.coverage import coverage
from formant_benchmark.evaluation.metrics.fdr import fdr
from formant_benchmark.evaluation.metrics.mae import mae
from formant_benchmark.evaluation.metrics.rmse import rmse

DEFAULT_METRICS = ("rmse", "mae", "avg", "fdr", "coverage")

__all__ = ["DEFAULT_METRICS", "avg", "coverage", "fdr", "mae", "rmse"]
