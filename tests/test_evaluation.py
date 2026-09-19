"""Phase 6 evaluation semantics across trajectory, static, and compatibility cases."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from formant_benchmark.data.models import (
    Formant,
    PredictionRun,
    PredictionRunManifest,
    TrackingInputMode,
)
from formant_benchmark.data.schemas import empty_failures
from formant_benchmark.evaluation.evaluator import evaluate
from formant_benchmark.exceptions import (
    DatasetFingerprintMismatchError,
    EvaluationCompatibilityError,
    IncompatibleFormantsError,
    UnsupportedScopeError,
)
from formant_benchmark.preparation.fingerprint import dataset_fingerprint
from tests.fixtures.synthetic import static_dataset, trajectory_dataset


def _prediction_run(
    dataset,
    predictions: pd.DataFrame,
    *,
    split: str | None = None,
    formants=None,
    input_mode: TrackingInputMode = TrackingInputMode.FULL_ITEM,
    interval_type: str | None = None,
) -> PredictionRun:
    tracker_formants = list(Formant if formants is None else formants)
    item_ids = dataset.items["item_id"].astype(str).tolist()
    if split is not None:
        selected = set(dataset.splits.loc[dataset.splits["split"] == split, "item_id"].astype(str))
        item_ids = [item_id for item_id in item_ids if item_id in selected]
        predictions = predictions.loc[predictions["item_id"].astype(str).isin(selected)].copy()
    parameters = pd.DataFrame(
        [{"item_id": item_id, "input_unit_id": item_id, "parameters_json": "{}"} for item_id in item_ids]
    )
    manifest = PredictionRunManifest(
        run_id="synthetic-run",
        status="completed",
        dataset_name=dataset.manifest.name,
        dataset_fingerprint=dataset_fingerprint(dataset),
        tracker="synthetic",
        tracker_formants=tracker_formants,
        prediction_formants=tracker_formants,
        input_mode=input_mode,
        interval_type=interval_type,
        split=split,
        configuration_digest="test",
        requested_inputs=len(parameters),
        succeeded_inputs=len(parameters),
        created_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    return PredictionRun(manifest, predictions, empty_failures(), parameters)


def test_all_scope_metrics_preserve_missing_prediction_semantics() -> None:
    dataset = trajectory_dataset()
    predictions = dataset.tracks.copy()
    predictions["F1"] += 10.0
    predictions["F2"] -= 20.0
    predictions["F3"] += 400.0
    predictions.loc[predictions["time_s"] == 0.2, "F4"] = np.nan

    result = evaluate(dataset, _prediction_run(dataset, predictions), scope="all")
    row = result.evaluation_unit_metrics.loc[result.evaluation_unit_metrics["item_id"] == "utt-1"].iloc[0]

    assert row["rmse_f1"] == pytest.approx(10.0)
    assert row["mae_f2"] == pytest.approx(20.0)
    assert row["avg_f1"] == pytest.approx(-10.0)
    assert row["avg_f2"] == pytest.approx(20.0)
    assert row["fdr_f1"] == pytest.approx(1.0)
    assert row["fdr_f3"] == pytest.approx(0.0)
    assert row["coverage_f4"] == pytest.approx(0.8)
    assert row["n_gold_measurements"] == 5
    assert row["n_predicted_measurements"] == 5
    assert row["rmse"] == pytest.approx((10.0 + 20.0 + 400.0 + 0.0) / 4.0)


def test_interpolation_does_not_extrapolate_or_bridge_missing_samples() -> None:
    dataset = trajectory_dataset()
    predictions = dataset.tracks.loc[
        (dataset.tracks["time_s"] >= 0.1) & (dataset.tracks["time_s"] <= 0.3)
    ].copy()
    predictions.loc[predictions["time_s"] == 0.2, "F1"] = np.nan

    result = evaluate(
        dataset,
        _prediction_run(dataset, predictions),
        scope="all",
        formants=["F1"],
    )
    row = result.evaluation_unit_metrics.loc[result.evaluation_unit_metrics["item_id"] == "utt-1"].iloc[0]
    assert row["rmse_f1"] == pytest.approx(0.0)
    assert row["coverage_f1"] == pytest.approx(2 / 5)
    assert row["fdr_f1"] == pytest.approx(2 / 5)


def test_central_vowel_region_and_joint_subgroups() -> None:
    dataset = trajectory_dataset()
    dataset.intervals = pd.concat(
        [
            dataset.intervals,
            pd.DataFrame(
                [{
                    "interval_id": "utt-2:v1",
                    "item_id": "utt-2",
                    "interval_type": "vowel",
                    "label": "aa",
                    "start_s": 0.1,
                    "end_s": 0.3,
                    "origin": "derived",
                }]
            ),
        ],
        ignore_index=True,
    )
    predictions = dataset.tracks.copy()
    run = _prediction_run(dataset, predictions)

    result = evaluate(
        dataset,
        run,
        scope="vowels",
        central_region=0.5,
        group_by=[["gender"], ["gender", "vowel"]],
    )

    assert len(result.evaluation_unit_metrics) == 4
    assert set(result.evaluation_unit_metrics["region"]) == {"full", "central_0.5"}
    central = result.evaluation_unit_metrics.loc[result.evaluation_unit_metrics["region"] == "central_0.5"]
    assert set(central["n_gold_measurements"]) == {1}
    assert set(result.evaluation_unit_metrics["vowel"]) == {"iy", "aa"}
    assert set(result.aggregate_metrics["grouping"]) == {"overall", "gender", "gender+vowel"}
    assert len(result.aggregate_metrics.loc[result.aggregate_metrics["grouping"] == "gender+vowel"]) == 4


def test_static_points_relative_positions_and_windows_are_evaluated_source_natively() -> None:
    dataset = static_dataset()
    rows: list[dict[str, object]] = []
    for item in dataset.items.itertuples(index=False):
        values = (700.0, 1700.0, 2700.0) if item.item_id == "vowel-1" else (300.0, 2300.0, 3100.0)
        for time_s in np.arange(0.0, float(item.duration_s) + 0.0001, 0.1):
            rows.append(
                {
                    "item_id": item.item_id,
                    "time_s": float(time_s),
                    "F1": values[0],
                    "F2": values[1],
                    "F3": values[2],
                    "F4": np.nan,
                }
            )
    predictions = pd.DataFrame(rows)

    result = evaluate(
        dataset,
        _prediction_run(dataset, predictions),
        scope="vowels",
        formants=["F1", "F2", "F3"],
        group_by=[["measurement_kind"]],
    )
    rows_by_id = result.evaluation_unit_metrics.set_index("measurement_id")

    assert rows_by_id.loc["m-1", "rmse_f1"] == pytest.approx(0.0)
    assert rows_by_id.loc["m-2", "rmse_f1"] == pytest.approx(10.0)
    assert rows_by_id.loc["m-2", "avg_f1"] == pytest.approx(10.0)
    assert rows_by_id.loc["m-3", "rmse_f1"] == pytest.approx(0.0)
    assert set(result.evaluation_unit_metrics["evaluation_unit_type"]) == {"static_measurement"}
    assert set(result.evaluation_unit_metrics["vowel"]) == {"ae", "iy"}


def test_static_midpoint_is_only_used_as_location_fallback() -> None:
    dataset = static_dataset()
    dataset.static_measurements = dataset.static_measurements.iloc[[0]].copy()
    dataset.static_measurements.loc[:, ["relative_position", "time_s", "window_start_s", "window_end_s"]] = np.nan
    prediction = pd.DataFrame(
        [
            {"item_id": "vowel-1", "time_s": 0.0, "F1": 600.0, "F2": 1600.0, "F3": 2600.0, "F4": np.nan},
            {"item_id": "vowel-1", "time_s": 0.5, "F1": 800.0, "F2": 1800.0, "F3": 2800.0, "F4": np.nan},
        ]
    )
    run = _prediction_run(dataset, prediction)

    result = evaluate(dataset, run, scope="all", formants=["F1", "F2", "F3"])
    row = result.evaluation_unit_metrics.iloc[0]
    assert row["rmse_f1"] == pytest.approx(0.0)
    assert row["rmse_f2"] == pytest.approx(0.0)
    assert row["rmse_f3"] == pytest.approx(0.0)


def test_formant_resolution_uses_declared_tracker_capability_and_validates_explicit_selection() -> None:
    dataset = trajectory_dataset()
    predictions = dataset.tracks.copy()
    predictions["F4"] = np.nan
    run = _prediction_run(dataset, predictions)
    run.manifest.prediction_formants = [Formant.F1, Formant.F2, Formant.F3]

    automatic = evaluate(dataset, run, scope="all")
    assert automatic.manifest.selected_formants == [Formant.F1, Formant.F2, Formant.F3]

    with pytest.raises(IncompatibleFormantsError, match="not available in prediction run"):
        evaluate(dataset, run, scope="all", formants=["F4"])


def test_voiced_scope_and_incompatible_fingerprint_fail_early() -> None:
    dataset = trajectory_dataset()
    run = _prediction_run(dataset, dataset.tracks.copy())
    with pytest.raises(UnsupportedScopeError, match="Voiced scope is not implemented yet"):
        evaluate(dataset, run, scope="voiced")

    run.manifest.dataset_fingerprint = "different"
    with pytest.raises(DatasetFingerprintMismatchError):
        evaluate(dataset, run, scope="all")


def test_fingerprint_mismatch_precedes_formant_compatibility() -> None:
    dataset = trajectory_dataset()
    run = _prediction_run(dataset, dataset.tracks.copy(), formants=[Formant.F1])
    run.manifest.prediction_formants = []
    run.manifest.dataset_fingerprint = "different"

    with pytest.raises(DatasetFingerprintMismatchError):
        evaluate(dataset, run, scope="all")


def test_all_scope_rejects_cropped_interval_prediction_run() -> None:
    dataset = trajectory_dataset()
    run = _prediction_run(
        dataset,
        dataset.tracks.copy(),
        input_mode=TrackingInputMode.CROPPED_INTERVALS,
        interval_type="vowel",
    )

    with pytest.raises(EvaluationCompatibilityError, match="cropped intervals.*cannot be evaluated with scope 'all'"):
        evaluate(dataset, run, scope="all")

    compatible = evaluate(dataset, run, scope="vowels")
    assert compatible.manifest.scope.value == "vowels"


def test_evaluation_split_defaults_to_prediction_run_split_and_rejects_mismatch() -> None:
    dataset = trajectory_dataset()
    run = _prediction_run(dataset, dataset.tracks.copy(), split="train")

    result = evaluate(dataset, run, scope="all")
    assert result.manifest.split == "train"
    assert result.evaluation_unit_metrics["item_id"].tolist() == ["utt-1"]

    with pytest.raises(EvaluationCompatibilityError, match="cannot be evaluated"):
        evaluate(dataset, run, scope="all", split="test")
