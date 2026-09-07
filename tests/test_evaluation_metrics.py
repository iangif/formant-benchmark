"""Focused numerical tests for the modular V1 metric implementations."""

import numpy as np
import pytest

from formant_benchmark.evaluation.metrics import avg, coverage, fdr, mae, rmse


def test_metrics_use_only_matched_values_but_fdr_and_coverage_keep_missing_in_denominator() -> None:
    gold = np.asarray([100.0, 200.0, 300.0, 400.0])
    prediction = np.asarray([110.0, np.nan, 600.0, 100.0])

    assert rmse(gold, prediction) == pytest.approx(np.sqrt((10**2 + 300**2 + 300**2) / 3))
    assert mae(gold, prediction) == pytest.approx((10 + 300 + 300) / 3)
    assert avg(gold, prediction) == pytest.approx((100 - 110 + 300 - 600 + 400 - 100) / 3)
    assert coverage(gold, prediction) == pytest.approx(0.75)
    assert fdr(gold, prediction, relative_threshold=0.30, absolute_threshold_hz=300.0) == pytest.approx(0.25)


def test_metrics_are_undefined_when_no_eligible_gold_exists() -> None:
    gold = np.asarray([np.nan, np.nan])
    prediction = np.asarray([100.0, np.nan])
    for value in (rmse(gold, prediction), mae(gold, prediction), avg(gold, prediction), coverage(gold, prediction), fdr(gold, prediction)):
        assert np.isnan(value)
