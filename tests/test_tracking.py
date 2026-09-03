"""Phase 5 tracker infrastructure and deterministic PredictionRun tests."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from formant_benchmark.data.models import TrackingInputMode
from formant_benchmark.exceptions import (
    ResumeCompatibilityError,
    UnsupportedVoicedFeatureError,
)
from formant_benchmark.execution.backends import LocalExecutionBackend
from formant_benchmark.execution.inputs import build_tracking_inputs
from formant_benchmark.preparation.fingerprint import dataset_fingerprint
from formant_benchmark.runs import runner as runner_module
from formant_benchmark.runs.io import load_prediction_run
from formant_benchmark.trackers.synthetic import SyntheticTracker
from tests.fixtures.synthetic import trajectory_dataset


def _inputs(dataset, tracker, tmp_path, *, mode=TrackingInputMode.FULL_ITEM, interval_type=None, split=None):
    return build_tracking_inputs(
        dataset,
        tracker,
        input_mode=mode,
        interval_type=interval_type,
        split=split,
        temporary_directory=tmp_path / "inputs",
    )


def _run(dataset, tracker, inputs, output, **kwargs):
    return tracker.run(
        inputs,
        config=kwargs.pop("config", {}),
        destination=output,
        dataset_fingerprint=dataset_fingerprint(dataset),
        dataset_name=dataset.manifest.name,
        input_mode=kwargs.pop("input_mode", TrackingInputMode.FULL_ITEM),
        **kwargs,
    )


def test_synthetic_tracker_writes_normalized_prediction_artifacts(tmp_path: Path) -> None:
    dataset = trajectory_dataset()
    tracker = SyntheticTracker()
    run = _run(
        dataset,
        tracker,
        _inputs(dataset, tracker, tmp_path),
        tmp_path / "run",
        cli_parameters={"frame_step_s": 0.1, "omit_formants": ["F4"]},
    )

    assert run.manifest.status == "completed"
    assert run.manifest.succeeded_inputs == 2
    assert [value.value for value in run.manifest.prediction_formants] == ["F1", "F2", "F3"]
    assert run.predictions.columns.tolist() == ["item_id", "time_s", "F1", "F2", "F3", "F4"]
    assert run.predictions["F4"].isna().all()
    assert {path.name for path in (tmp_path / "run").iterdir()} == {
        "run_manifest.yaml", "predictions.parquet", "failures.parquet", "item_parameters.parquet"
    }
    load_prediction_run(tmp_path / "run")


def test_metadata_overrides_and_failures_are_persisted_per_input(tmp_path: Path) -> None:
    dataset = trajectory_dataset()
    tracker = SyntheticTracker()
    overrides = {
        "overrides": [
            {"where": {"gender": "female"}, "parameters": {"offset_hz": 100}},
            {"where": {"item_id": "utt-2"}, "parameters": {"fail": True}},
        ]
    }
    run = _run(
        dataset,
        tracker,
        _inputs(dataset, tracker, tmp_path),
        tmp_path / "run",
        dataset_tracker_config=overrides,
        cli_parameters={"frame_step_s": 0.2},
    )

    assert run.manifest.succeeded_inputs == 1
    assert run.manifest.failed_inputs == 1
    assert run.failures.iloc[0]["failure_type"] == "execution_failed"
    assert run.predictions.loc[run.predictions["item_id"] == "utt-1", "F1"].iloc[0] == 600.0
    parameters = {
        row.input_unit_id: json.loads(row.parameters_json)
        for row in run.item_parameters.itertuples(index=False)
    }
    assert parameters["utt-1"]["offset_hz"] == 100
    assert parameters["utt-2"]["fail"] is True


def test_item_failure_does_not_restart_persistent_process(tmp_path: Path, monkeypatch) -> None:
    starts = 0
    original = LocalExecutionBackend.start_worker

    def counted_start(self, command, work_root):
        nonlocal starts
        starts += 1
        return original(self, command, work_root)

    monkeypatch.setattr(LocalExecutionBackend, "start_worker", counted_start)
    dataset = trajectory_dataset()
    tracker = SyntheticTracker()
    run = _run(
        dataset,
        tracker,
        _inputs(dataset, tracker, tmp_path),
        tmp_path / "run",
        dataset_tracker_config={
            "overrides": [{"where": {"item_id": "utt-1"}, "parameters": {"fail": True}}]
        },
    )

    assert starts == 1
    assert run.manifest.failed_inputs == 1
    assert run.manifest.succeeded_inputs == 1
    assert run.predictions["item_id"].unique().tolist() == ["utt-2"]


def test_crashed_worker_is_restarted_for_the_next_input(tmp_path: Path, monkeypatch) -> None:
    starts = 0
    original = LocalExecutionBackend.start_worker

    def counted_start(self, command, work_root):
        nonlocal starts
        starts += 1
        return original(self, command, work_root)

    monkeypatch.setattr(LocalExecutionBackend, "start_worker", counted_start)
    dataset = trajectory_dataset()
    tracker = SyntheticTracker()
    run = _run(
        dataset,
        tracker,
        _inputs(dataset, tracker, tmp_path),
        tmp_path / "run",
        dataset_tracker_config={
            "overrides": [{"where": {"item_id": "utt-1"}, "parameters": {"crash": True}}]
        },
    )

    assert starts == 2
    assert run.manifest.failed_inputs == 1
    assert run.manifest.succeeded_inputs == 1
    assert run.predictions["item_id"].unique().tolist() == ["utt-2"]


def test_timed_out_worker_is_restarted_for_the_next_input(tmp_path: Path, monkeypatch) -> None:
    starts = 0
    original = LocalExecutionBackend.start_worker

    def counted_start(self, command, work_root):
        nonlocal starts
        starts += 1
        return original(self, command, work_root)

    monkeypatch.setattr(LocalExecutionBackend, "start_worker", counted_start)
    dataset = trajectory_dataset()
    tracker = SyntheticTracker()
    run = _run(
        dataset,
        tracker,
        _inputs(dataset, tracker, tmp_path),
        tmp_path / "run",
        config={"execution": {"timeout_s": 0.05}},
        dataset_tracker_config={
            "overrides": [{"where": {"item_id": "utt-1"}, "parameters": {"hang_s": 1}}]
        },
    )

    assert starts == 2
    assert run.failures.iloc[0]["failure_type"] == "timeout"
    assert run.predictions["item_id"].unique().tolist() == ["utt-2"]


def test_max_items_per_process_recycles_worker(tmp_path: Path, monkeypatch) -> None:
    starts = 0
    original = LocalExecutionBackend.start_worker

    def counted_start(self, command, work_root):
        nonlocal starts
        starts += 1
        return original(self, command, work_root)

    monkeypatch.setattr(LocalExecutionBackend, "start_worker", counted_start)
    dataset = trajectory_dataset()
    tracker = SyntheticTracker()
    _run(
        dataset,
        tracker,
        _inputs(dataset, tracker, tmp_path),
        tmp_path / "run",
        config={"execution": {"max_items_per_process": 1}},
    )

    assert starts == 2


def test_results_are_checkpointed_in_batches(tmp_path: Path, monkeypatch) -> None:
    writes = 0
    original = runner_module.write_prediction_run

    def counted_write(run, destination):
        nonlocal writes
        writes += 1
        return original(run, destination)

    monkeypatch.setattr(runner_module, "write_prediction_run", counted_write)
    dataset = trajectory_dataset()
    tracker = SyntheticTracker()
    _run(dataset, tracker, _inputs(dataset, tracker, tmp_path), tmp_path / "run")

    assert writes == 3  # initial, one batched checkpoint, and completed manifest


def test_generic_runner_reports_progress_for_every_input(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, object] = {"updates": 0, "closed": False}

    class FakeProgress:
        def update(self, amount=1):
            observed["updates"] = int(observed["updates"]) + amount

        def close(self):
            observed["closed"] = True

    def fake_tqdm(**kwargs):
        observed.update(kwargs)
        return FakeProgress()

    monkeypatch.setattr(runner_module, "tqdm", fake_tqdm)
    dataset = trajectory_dataset()
    tracker = SyntheticTracker()
    _run(
        dataset,
        tracker,
        _inputs(dataset, tracker, tmp_path),
        tmp_path / "run",
        show_progress=True,
    )

    assert observed["total"] == 2
    assert observed["initial"] == 0
    assert observed["desc"] == "Tracking synthetic"
    assert observed["unit"] == "input"
    assert observed["updates"] == 2
    assert observed["closed"] is True


def test_resume_requires_identical_configuration(tmp_path: Path) -> None:
    dataset = trajectory_dataset()
    tracker = SyntheticTracker()
    inputs = _inputs(dataset, tracker, tmp_path)
    output = tmp_path / "run"
    original = _run(dataset, tracker, inputs, output, cli_parameters={"frame_step_s": 0.2})
    resumed = _run(dataset, tracker, inputs, output, cli_parameters={"frame_step_s": 0.2}, resume=True)
    assert len(resumed.predictions) == len(original.predictions)

    with pytest.raises(ResumeCompatibilityError):
        _run(dataset, tracker, inputs, output, cli_parameters={"frame_step_s": 0.1}, resume=True)


def test_cropped_interval_times_are_returned_to_parent_coordinates(tmp_path: Path) -> None:
    dataset = trajectory_dataset()
    audio = tmp_path / "utt-1.wav"
    _write_wav(audio, duration_s=0.4)
    dataset.items.loc[dataset.items["item_id"] == "utt-1", "audio_path"] = str(audio)
    tracker = SyntheticTracker()
    inputs = _inputs(
        dataset,
        tracker,
        tmp_path,
        mode=TrackingInputMode.CROPPED_INTERVALS,
        interval_type="vowel",
        split="train",
    )
    run = _run(
        dataset,
        tracker,
        inputs,
        tmp_path / "run",
        input_mode=TrackingInputMode.CROPPED_INTERVALS,
        interval_type="vowel",
        split="train",
        cli_parameters={"frame_step_s": 0.1},
    )
    assert run.item_parameters["input_unit_id"].tolist() == ["utt-1:v1"]
    assert run.predictions["time_s"].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_voiced_input_fails_before_execution(tmp_path: Path) -> None:
    dataset = trajectory_dataset()
    with pytest.raises(UnsupportedVoicedFeatureError):
        _inputs(
            dataset,
            SyntheticTracker(),
            tmp_path,
            mode=TrackingInputMode.CROPPED_INTERVALS,
            interval_type="voiced",
        )


def _write_wav(path: Path, *, duration_s: float, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * round(duration_s * sample_rate))
