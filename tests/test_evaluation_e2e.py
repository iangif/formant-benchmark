"""Deterministic end-to-end Phase 6 benchmarks using the test-only tracker."""

from pathlib import Path

import pytest

from formant_benchmark.data.models import TrackingInputMode
from formant_benchmark.evaluation import evaluate, load_evaluation_result, write_evaluation_result
from formant_benchmark.execution.inputs import build_tracking_inputs
from formant_benchmark.preparation.fingerprint import dataset_fingerprint
from formant_benchmark.trackers.synthetic import SyntheticTracker
from tests.fixtures.synthetic import static_dataset, trajectory_dataset


def _track(dataset, tmp_path: Path, *, overrides=None, cli_parameters=None):
    tracker = SyntheticTracker()
    inputs = build_tracking_inputs(
        dataset,
        tracker,
        input_mode=TrackingInputMode.FULL_ITEM,
        interval_type=None,
        split=None,
        temporary_directory=tmp_path / "inputs",
    )
    return tracker.run(
        inputs,
        config={},
        destination=tmp_path / "run",
        dataset_fingerprint=dataset_fingerprint(dataset),
        dataset_name=dataset.manifest.name,
        dataset_tracker_config=overrides,
        cli_parameters=cli_parameters or {},
        input_mode=TrackingInputMode.FULL_ITEM,
        show_progress=False,
    )


def test_synthetic_trajectory_track_evaluate_persist_round_trip(tmp_path: Path) -> None:
    dataset = trajectory_dataset()
    run = _track(
        dataset,
        tmp_path,
        overrides={
            "overrides": [
                {"where": {"item_id": "utt-2"}, "parameters": {"offset_hz": 10.0}},
            ]
        },
        cli_parameters={"frame_step_s": 0.1},
    )

    result = evaluate(dataset, run, scope="all", group_by=[["gender"]], evaluation_id="exact")
    assert result.evaluation_unit_metrics["rmse"].tolist() == pytest.approx([0.0, 0.0])
    assert result.evaluation_unit_metrics["coverage"].tolist() == pytest.approx([1.0, 1.0])

    destination = tmp_path / "evaluation"
    write_evaluation_result(result, destination)
    loaded = load_evaluation_result(destination)
    assert loaded.manifest.selected_formants == result.manifest.selected_formants
    assert set(path.name for path in destination.iterdir()) == {
        "evaluation_manifest.yaml",
        "evaluation_unit_metrics.parquet",
        "aggregate_metrics.parquet",
    }


def test_synthetic_static_track_produces_known_point_and_window_metrics(tmp_path: Path) -> None:
    dataset = static_dataset()
    run = _track(
        dataset,
        tmp_path,
        overrides={
            "parameters": {
                "base_f1": 700.0,
                "base_f2": 1700.0,
                "base_f3": 2700.0,
                "slope_hz_per_s": 0.0,
                "omit_formants": ["F4"],
            },
            "overrides": [
                {
                    "where": {"item_id": "vowel-2"},
                    "parameters": {"base_f1": 300.0, "base_f2": 2300.0, "base_f3": 3100.0},
                }
            ],
        },
        cli_parameters={"frame_step_s": 0.1},
    )

    result = evaluate(dataset, run, scope="vowels", evaluation_id="static-known")
    by_measurement = result.evaluation_unit_metrics.set_index("measurement_id")
    assert by_measurement.loc["m-1", "rmse_f1"] == pytest.approx(0.0)
    assert by_measurement.loc["m-2", "rmse_f1"] == pytest.approx(10.0)
    assert by_measurement.loc["m-3", "rmse_f1"] == pytest.approx(0.0)
