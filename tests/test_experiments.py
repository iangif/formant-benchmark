"""Experiment configs, name-or-path lookup, run/evaluation reuse, and CLI defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from formant_benchmark.cli.main import main
from formant_benchmark.config.discovery import (
    load_tracker_config,
    resolve_config_file,
    resolve_prepared_dataset,
)
from formant_benchmark.data.io import load_prepared_dataset, write_prepared_dataset
from formant_benchmark.data.models import TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError, ExperimentConflictError, UnsupportedScopeError
from formant_benchmark.experiments import load_experiment_config, run_experiment
from formant_benchmark.runs import read_prediction_run_manifest
from tests.fixtures.hillenbrand import create_hillenbrand_layout
from tests.fixtures.synthetic import trajectory_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_EXPERIMENT = {
    "dataset": "synthetic_trajectory",
    "tracker": "synthetic",
    "tracking": {"input_mode": "full_item", "split": "train", "parameters": {"frame_step_s": 0.1}},
    "evaluation": {"scope": "vowels"},
}


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch) -> Path:
    """A working directory laid out like the repository, with one prepared dataset."""
    monkeypatch.chdir(tmp_path)
    write_prepared_dataset(trajectory_dataset(), tmp_path / "prepared" / "synthetic_trajectory")
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    return tmp_path


def _write_experiment(root: Path, name: str, **changes: object) -> Path:
    config = yaml.safe_load(yaml.safe_dump(_EXPERIMENT))
    for section, values in changes.items():
        if isinstance(values, dict):
            config[section].update(values)
        else:
            config[section] = values
    path = root / "configs" / "experiments" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(name: str, **options: object):
    experiment = load_experiment_config(resolve_config_file(name, "experiments"))
    return run_experiment(
        experiment,
        dataset=load_prepared_dataset(resolve_prepared_dataset(experiment.dataset)),
        tracker_config=load_tracker_config(experiment.tracker_config, experiment.tracker),
        **options,
    )


def test_experiment_config_defaults_and_validation(tmp_path: Path) -> None:
    path = _write_experiment_file(tmp_path / "vowel_check.yaml", _EXPERIMENT)
    experiment = load_experiment_config(path)
    assert experiment.name == "vowel_check"
    assert experiment.tracking.input_mode is TrackingInputMode.FULL_ITEM
    assert experiment.tracking.interval_padding_s == 0.0
    assert experiment.evaluation_split == "train"
    assert experiment.tracking.parameter_layer() == {"parameters": {"frame_step_s": 0.1}}

    typo = dict(_EXPERIMENT, tracking={**_EXPERIMENT["tracking"], "interval_typ": "vowel"})
    with pytest.raises(ConfigurationError, match="tracking.interval_typ"):
        load_experiment_config(_write_experiment_file(tmp_path / "typo.yaml", typo))
    with pytest.raises(ConfigurationError, match="plain name"):
        load_experiment_config(_write_experiment_file(tmp_path / "bad.yaml", dict(_EXPERIMENT, name="a/b")))


def test_template_is_a_valid_experiment() -> None:
    experiment = load_experiment_config(REPOSITORY_ROOT / "configs" / "experiments" / "template.yaml")
    assert experiment.name == "template"
    assert experiment.tracker == "praat_burg"
    assert experiment.evaluation_split is None


def test_names_resolve_to_conventional_locations(workspace: Path) -> None:
    experiment = _write_experiment(workspace, "demo")
    assert resolve_config_file("demo", "experiments") == Path("configs/experiments/demo.yaml")
    assert resolve_config_file(experiment, "experiments") == experiment
    assert resolve_prepared_dataset("synthetic_trajectory") == Path("prepared/synthetic_trajectory")
    with pytest.raises(ConfigurationError, match="prepared/missing does not exist"):
        resolve_prepared_dataset("missing")
    with pytest.raises(ConfigurationError, match="configs/trackers/missing.yaml does not exist"):
        resolve_config_file("missing", "trackers")

    assert load_tracker_config(None, "synthetic") == {}
    trackers = workspace / "configs" / "trackers"
    trackers.mkdir()
    (trackers / "synthetic.yaml").write_text("tracker: synthetic\nparameters: {offset_hz: 5.0}\n")
    assert load_tracker_config(None, "synthetic")["parameters"] == {"offset_hz": 5.0}
    with pytest.raises(ConfigurationError, match="declares 'synthetic'"):
        load_tracker_config("synthetic", "praat_burg")


def test_experiment_tracks_then_reuses_run_and_evaluation(workspace: Path) -> None:
    _write_experiment(workspace, "first")
    first = _run("first")
    assert (first.run_action, first.evaluation_action) == ("tracked", "evaluated")
    assert first.run.root == Path("runs/first")
    assert first.evaluation.root == Path("evaluations/first")
    assert first.evaluation.manifest.settings_digest

    again = _run("first")
    assert (again.run_action, again.evaluation_action) == ("reused", "reused")

    # Same tracking under another name shares the run but gets its own evaluation.
    _write_experiment(workspace, "central", evaluation={"central_region": 0.5})
    central = _run("central")
    assert (central.run_action, central.evaluation_action) == ("reused", "evaluated")
    assert central.run.root == Path("runs/first")
    assert central.evaluation.manifest.prediction_run_id == "first"
    assert not Path("runs/central").exists()


def test_experiment_refuses_mismatched_outputs_and_overwrites_on_request(workspace: Path) -> None:
    _write_experiment(workspace, "demo")
    _run("demo")

    _write_experiment(workspace, "demo", tracking={"parameters": {"frame_step_s": 0.2}})
    with pytest.raises(ExperimentConflictError, match="different tracking condition"):
        _run("demo")

    _write_experiment(workspace, "demo", evaluation={"formants": ["F1", "F2"]})
    with pytest.raises(ExperimentConflictError, match="--overwrite"):
        _run("demo")
    outcome = _run("demo", overwrite=True)
    assert (outcome.run_action, outcome.evaluation_action) == ("reused", "overwritten")
    assert [f.value for f in outcome.evaluation.manifest.selected_formants] == ["F1", "F2"]


def test_unfinished_run_requires_explicit_resume(workspace: Path) -> None:
    _write_experiment(workspace, "demo")
    _run("demo")
    manifest_path = Path("runs/demo/run_manifest.yaml")
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["status"] = "running"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ExperimentConflictError, match="--resume"):
        _run("demo")
    outcome = _run("demo", resume=True)
    assert outcome.run_action == "resumed"
    assert read_prediction_run_manifest("runs/demo").status == "completed"


def test_override_fields_are_checked_before_tracking(workspace: Path, capsys) -> None:
    rule = {"where": {"gendr": "female"}, "parameters": {"offset_hz": 5.0}}
    _write_experiment(workspace, "typo", tracking={"overrides": [rule]})
    with pytest.raises(ConfigurationError, match="'gendr' that no tracking input has"):
        _run("typo")
    assert not Path("runs").exists()

    rule = {"where": {"gender": "Female"}, "parameters": {"offset_hz": 5.0}}
    _write_experiment(workspace, "nomatch", tracking={"overrides": [rule]})
    assert main(["experiment", "run", "nomatch"]) == 0
    tracking = yaml.safe_load(capsys.readouterr().out)["tracking"]
    assert tracking["warnings"] == [
        "override 0 in the experiment config (gender='Female') matches no tracking input"
    ]
    assert read_prediction_run_manifest("runs/nomatch").warnings == tracking["warnings"]


def test_unimplemented_scope_fails_before_tracking(workspace: Path) -> None:
    _write_experiment(workspace, "voiced", evaluation={"scope": "voiced"})
    with pytest.raises(UnsupportedScopeError):
        _run("voiced")
    assert not Path("runs").exists()


def test_experiment_run_cli(workspace: Path, capsys) -> None:
    _write_experiment(workspace, "demo")
    assert main(["experiment", "run", "demo"]) == 0
    first = yaml.safe_load(capsys.readouterr().out)
    assert first["tracking"] == {"run": "runs/demo", "action": "tracked"}
    assert first["evaluation"] == {"output": "evaluations/demo", "action": "evaluated"}

    assert main(["experiment", "run", "demo"]) == 0
    second = yaml.safe_load(capsys.readouterr().out)
    assert second["tracking"]["action"] == "reused"
    assert second["evaluation"]["action"] == "reused"

    # A CLI parameter changes the tracking condition, which runs/demo does not match.
    assert main(["experiment", "run", "demo", "--parameter", "offset_hz=10"]) == 2
    assert "different tracking condition" in capsys.readouterr().err


def test_track_and_evaluate_default_outputs_and_dataset(workspace: Path, capsys) -> None:
    assert main([
        "track", "--dataset", "synthetic_trajectory", "--tracker", "synthetic",
        "--input-mode", "full_item", "--split", "train", "--parameter", "frame_step_s=0.1",
    ]) == 0
    capsys.readouterr()
    [run] = list(Path("runs").iterdir())
    assert run.name.endswith("-synthetic_trajectory-synthetic")

    assert main(["evaluate", "--predictions", str(run), "--scope", "all"]) == 0
    capsys.readouterr()
    [evaluation] = list(Path("evaluations").iterdir())
    assert evaluation.name.endswith(f"-{run.name}-all")

    assert main(["dataset", "inspect", "synthetic_trajectory"]) == 0
    assert "synthetic_trajectory" in capsys.readouterr().out


def test_dataset_prepare_by_config_name_defaults_output(workspace: Path, capsys) -> None:
    root = create_hillenbrand_layout(workspace / "raw")
    datasets = workspace / "configs" / "datasets"
    datasets.mkdir()
    (datasets / "hb.yaml").write_text(
        yaml.safe_dump({"adapter": "hillenbrand", "name": "hillenbrand", "root": str(root), "language": "english"})
    )
    assert main(["dataset", "prepare", "--config", "hb"]) == 0
    capsys.readouterr()
    assert (workspace / "prepared" / "hillenbrand" / "dataset.yaml").is_file()


def _write_experiment_file(path: Path, config: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
