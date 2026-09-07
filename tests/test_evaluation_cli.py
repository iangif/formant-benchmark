"""CLI coverage for the Phase 6 evaluate command."""

from pathlib import Path

from formant_benchmark.cli.main import main
from formant_benchmark.data.io import write_prepared_dataset
from formant_benchmark.data.models import TrackingInputMode
from formant_benchmark.execution.inputs import build_tracking_inputs
from formant_benchmark.preparation.fingerprint import dataset_fingerprint
from formant_benchmark.trackers.synthetic import SyntheticTracker
from tests.fixtures.synthetic import trajectory_dataset


def test_evaluate_cli_writes_wide_outputs_and_resolved_manifest(tmp_path: Path, capsys) -> None:
    dataset = trajectory_dataset()
    dataset_path = tmp_path / "dataset"
    persisted = write_prepared_dataset(dataset, dataset_path)
    tracker = SyntheticTracker()
    inputs = build_tracking_inputs(
        persisted,
        tracker,
        input_mode=TrackingInputMode.FULL_ITEM,
        interval_type=None,
        split="train",
        temporary_directory=tmp_path / "inputs",
    )
    run_path = tmp_path / "run"
    tracker.run(
        inputs,
        config={},
        destination=run_path,
        dataset_fingerprint=persisted.manifest.fingerprint or dataset_fingerprint(persisted),
        dataset_name=persisted.manifest.name,
        input_mode=TrackingInputMode.FULL_ITEM,
        split="train",
        cli_parameters={"frame_step_s": 0.1},
        show_progress=False,
    )

    output = tmp_path / "evaluation"
    assert main([
        "evaluate",
        "--dataset", str(dataset_path),
        "--predictions", str(run_path),
        "--output", str(output),
        "--scope", "vowels",
        "--formants", "F1", "F2", "F3",
        "--central-region", "0.5",
        "--group-by", "gender,vowel",
    ]) == 0
    stdout = capsys.readouterr().out
    assert "selected_formants:" in stdout
    assert "n_evaluation_units: 2" in stdout
    assert (output / "evaluation_manifest.yaml").is_file()
    assert (output / "evaluation_unit_metrics.parquet").is_file()
    assert (output / "aggregate_metrics.parquet").is_file()

    assert main([
        "evaluate",
        "--dataset", str(dataset_path),
        "--predictions", str(run_path),
        "--output", str(tmp_path / "voiced-evaluation"),
        "--scope", "voiced",
    ]) == 2
    assert "Voiced scope is not implemented yet" in capsys.readouterr().err
