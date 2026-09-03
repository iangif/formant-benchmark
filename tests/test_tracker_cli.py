"""CLI coverage for tracker discovery, checks, and synthetic execution."""

from pathlib import Path

from formant_benchmark.cli.main import main
from formant_benchmark.data.io import write_prepared_dataset
from tests.fixtures.synthetic import trajectory_dataset


def test_tracker_list_inspect_and_check(capsys) -> None:
    assert main(["tracker", "list"]) == 0
    assert capsys.readouterr().out.strip() == "synthetic"
    assert main(["tracker", "inspect", "synthetic"]) == 0
    assert "cropped_intervals" in capsys.readouterr().out
    assert main(["tracker", "check", "synthetic"]) == 0
    assert "available: true" in capsys.readouterr().out


def test_track_cli_creates_prediction_run(tmp_path: Path, capsys) -> None:
    dataset_path = tmp_path / "dataset"
    write_prepared_dataset(trajectory_dataset(), dataset_path)
    run_path = tmp_path / "run"
    assert main([
        "track",
        "--dataset", str(dataset_path),
        "--tracker", "synthetic",
        "--output", str(run_path),
        "--input-mode", "full_item",
        "--split", "train",
        "--parameter", "frame_step_s=0.2",
    ]) == 0
    output = capsys.readouterr().out
    assert "succeeded_inputs: 1" in output
    assert (run_path / "predictions.parquet").is_file()
