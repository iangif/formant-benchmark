"""Analysis loaders and the report writer over small persisted benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from formant_benchmark.analysis.load import load_unit_metrics, speaker_bootstrap_ci
from formant_benchmark.analysis.positions import load_position_errors
from formant_benchmark.data.io import write_prepared_dataset
from formant_benchmark.evaluation import evaluate, write_evaluation_result
from formant_benchmark.exceptions import EvaluationValidationError
from formant_benchmark.runs.io import write_prediction_run
from tests.fixtures.synthetic import static_dataset, trajectory_dataset
from tests.test_evaluation import _prediction_run


def _persist(root: Path, dataset, predictions: pd.DataFrame, name: str) -> None:
    write_prepared_dataset(dataset, root / "prepared" / dataset.manifest.name)
    run = _prediction_run(dataset, predictions)
    run.manifest.run_id = f"{name}-run"
    write_prediction_run(run, root / "runs" / f"{name}-run")
    write_evaluation_result(evaluate(dataset, run, scope="vowels", evaluation_id=name), root / "evaluations" / name)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    trajectory = trajectory_dataset()
    predictions = trajectory.tracks.copy()
    predictions["F1"] += 10.0
    _persist(tmp_path, trajectory, predictions, "trajectory")

    static = static_dataset()
    rows = [
        {"item_id": item_id, "time_s": time_s, "F1": 690.0, "F2": 1700.0, "F3": 2700.0, "F4": np.nan}
        for item_id in static.items["item_id"]
        for time_s in np.arange(0.0, 0.61, 0.1)
    ]
    _persist(tmp_path, static, pd.DataFrame(rows), "static")
    return tmp_path


def test_unit_metrics_are_long_by_formant_and_exclude_f4(workspace: Path) -> None:
    units = load_unit_metrics(workspace / "evaluations")
    trajectory = units.loc[units["evaluation"] == "trajectory"]

    assert set(units["formant"]) == {"F1", "F2", "F3"}
    assert set(units["dataset"]) == {"synthetic_trajectory", "synthetic_static"}
    assert trajectory.loc[trajectory["formant"] == "F1", "mae"].tolist() == pytest.approx([10.0])
    row = trajectory.loc[trajectory["formant"] == "F1"].iloc[0]
    assert row["pred_mean"] - row["gold_mean"] == pytest.approx(10.0)
    assert {"speaker_id", "vowel", "tracker", "prediction_run_id"} <= set(units.columns)
    assert "mae_f1" not in units.columns


def test_schema_1_evaluations_are_rejected(workspace: Path) -> None:
    manifest = workspace / "evaluations" / "static" / "evaluation_manifest.yaml"
    manifest.write_text(manifest.read_text().replace("schema_version: '2'", "schema_version: '1'"))
    with pytest.raises(EvaluationValidationError, match="--overwrite"):
        load_unit_metrics(workspace / "evaluations")


def test_speaker_bootstrap_resamples_whole_speakers() -> None:
    values = pd.Series([1.0, 1.0, 1.0, 5.0])
    single = speaker_bootstrap_ci(values, pd.Series(["a"] * 4))
    assert single == pytest.approx((2.0, 2.0, 2.0))

    mean, low, high = speaker_bootstrap_ci(values, pd.Series(["a", "a", "a", "b"]), seed=1)
    assert mean == pytest.approx(2.0)
    assert 1.0 <= low <= mean <= high <= 5.0


def test_position_errors_bin_trajectory_frames_and_keep_static_positions(workspace: Path) -> None:
    units = load_unit_metrics(workspace / "evaluations")
    positions = load_position_errors(units, prepared_root=workspace / "prepared", runs_root=workspace / "runs")

    trajectory = positions.loc[(positions["dataset"] == "synthetic_trajectory") & (positions["formant"] == "F1")]
    # The vowel spans 0.1-0.3 s with gold frames at its start, middle, and end.
    assert sorted(trajectory["position"]) == pytest.approx([0.05, 0.55, 0.95])
    assert trajectory["error"].tolist() == pytest.approx([10.0, 10.0, 10.0])

    static = positions.loc[(positions["dataset"] == "synthetic_static") & (positions["formant"] == "F1")]
    # Only the relative-position measurement has a position; point and window ones drop out.
    assert static["position"].tolist() == pytest.approx([0.5])
    assert static["error"].tolist() == pytest.approx([690.0 - 700.0])
    assert static["abs_error"].tolist() == pytest.approx([10.0])


def test_report_writes_every_figure_and_summary(workspace: Path) -> None:
    pytest.importorskip("matplotlib")
    from formant_benchmark.analysis.report import write_report

    written = write_report(
        evaluations_root=workspace / "evaluations",
        prepared_root=workspace / "prepared",
        runs_root=workspace / "runs",
        output=workspace / "report",
        formats=("png",),
    )

    names = {path.name for path in written}
    assert {"heatmap_mae.png", "dots_mae.png", "ecdf_mae.png", "position_mae.png", "summary.csv"} <= names
    assert "vowel_space_synthetic_trajectory.png" in names
    summary = pd.read_csv(workspace / "report" / "summary.csv")
    f1 = summary.loc[(summary["dataset"] == "synthetic_trajectory") & (summary["formant"] == "F1")].iloc[0]
    assert f1["mae"] == pytest.approx(10.0)
    assert f1["mae_ci_low"] == pytest.approx(10.0)


def test_report_cli(workspace: Path, capsys) -> None:
    pytest.importorskip("matplotlib")
    from formant_benchmark.cli.main import main

    code = main(
        [
            "report",
            "--evaluations", str(workspace / "evaluations"),
            "--prepared", str(workspace / "prepared"),
            "--runs", str(workspace / "runs"),
            "--output", str(workspace / "cli-report"),
            "--format", "png",
            "--skip-positions",
        ]
    )

    assert code == 0
    assert (workspace / "cli-report" / "heatmap_mae.png").is_file()
    assert not (workspace / "cli-report" / "position_mae.png").exists()
    assert "summary.csv" in capsys.readouterr().out


def test_examples_write_a_pdf_and_listing_per_dataset(workspace: Path) -> None:
    pytest.importorskip("matplotlib")
    from formant_benchmark.analysis.examples import write_examples

    written = write_examples(
        evaluations_root=workspace / "evaluations",
        prepared_root=workspace / "prepared",
        runs_root=workspace / "runs",
        output=workspace / "examples",
        count=5,
    )

    names = {path.name for path in written}
    assert names == {
        "examples_synthetic_trajectory.pdf", "examples_synthetic_trajectory.csv",
        "examples_synthetic_static.pdf", "examples_synthetic_static.csv",
    }
    listing = pd.read_csv(workspace / "examples" / "examples_synthetic_trajectory.csv")
    # One tracker per fixture dataset: random, worst for that tracker, no disagreement section.
    assert listing["section"].unique().tolist() == ["Random", "Worst synthetic"]
    # The fixture's predictions are 10 Hz off on F1 only, so the F1-F3 mean is 10 / 3.
    assert listing["mae_synthetic"].tolist() == pytest.approx([10 / 3, 10 / 3])


def test_examples_draw_a_spectrogram_from_the_wav(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import wave

    import matplotlib.pyplot as plt
    from matplotlib.collections import QuadMesh

    from formant_benchmark.analysis.examples import _draw_spectrogram

    rate = 16000
    tone = (8000 * np.sin(2 * np.pi * 500 * np.arange(rate // 2) / rate)).astype(np.int16)
    audio = tmp_path / "tone.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(tone.tobytes())

    fig, ax = plt.subplots()
    _draw_spectrogram(ax, audio, (0.1, 0.3), 0.15, top=4000.0)
    _draw_spectrogram(ax, tmp_path / "missing.wav", (0.1, 0.3), 0.15, top=4000.0)
    meshes = [child for child in ax.get_children() if isinstance(child, QuadMesh)]
    assert len(meshes) == 1
    # Times are in ms from the vowel start, so the 0.1-0.3 s window spans about -50 to 150 ms.
    left, right = ax.get_xlim()
    assert -50 <= left < -40 and 140 < right <= 150
    plt.close(fig)


def test_examples_cli(workspace: Path, capsys) -> None:
    pytest.importorskip("matplotlib")
    from formant_benchmark.cli.main import main

    code = main(
        [
            "examples",
            "--evaluations", str(workspace / "evaluations"),
            "--prepared", str(workspace / "prepared"),
            "--runs", str(workspace / "runs"),
            "--output", str(workspace / "cli-examples"),
            "--dataset", "synthetic_static",
            "--count", "3",
            "--spectrograms", "none",
        ]
    )

    assert code == 0
    assert (workspace / "cli-examples" / "examples_synthetic_static.pdf").is_file()
    assert not (workspace / "cli-examples" / "examples_synthetic_trajectory.pdf").exists()
    assert "examples_synthetic_static.csv" in capsys.readouterr().out
