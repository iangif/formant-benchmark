"""Shared Python-wrapper command, health-check, and parameter-validation helpers."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.execution.backends import ContainerExecutionBackend
from formant_benchmark.trackers.base import TrackerAdapter, TrackerCapabilities
from formant_benchmark.trackers.parameters import nonnegative_float, positive_float, positive_int
from formant_benchmark.trackers.python_wrapper import python_wrapper_command, run_wrapper_check

# Prints the --check arguments it received plus a version chosen through environment
# variables, then exits with the requested status, mimicking a tracker wrapper.
_FAKE_WRAPPER = """
import json, os, sys
payload = {"arguments": sys.argv[1:]}
if os.environ.get("FAKE_VERSION"):
    payload["fake_version"] = os.environ["FAKE_VERSION"]
if os.environ.get("FAKE_ERROR"):
    payload["error"] = os.environ["FAKE_ERROR"]
print(json.dumps(payload))
raise SystemExit(int(os.environ.get("FAKE_EXIT", "0")))
"""


class _FakeTracker(TrackerAdapter):
    name = "fake"
    capabilities = TrackerCapabilities(
        formants=frozenset({Formant.F1}),
        input_modes=frozenset({TrackingInputMode.FULL_ITEM}),
    )

    def __init__(self, script: Path) -> None:
        self.script = script

    def wrapper_command(self, config: Mapping[str, Any]) -> Sequence[str]:
        return (str(config["execution"]["command"]), str(self.script))


def _config(**environment: str) -> dict[str, Any]:
    return {"execution": {"backend": "local", "command": sys.executable, "environment": environment}}


@pytest.fixture
def tracker(tmp_path: Path) -> _FakeTracker:
    script = tmp_path / "fake_wrapper.py"
    script.write_text(_FAKE_WRAPPER, encoding="utf-8")
    return _FakeTracker(script)


def test_python_wrapper_command_accepts_string_or_list_and_requires_command() -> None:
    assert python_wrapper_command(
        {"execution": {"command": "/env/bin/python"}}, module="pkg.wrapper", label="X", environment="x"
    ) == ("/env/bin/python", "-m", "pkg.wrapper")
    assert python_wrapper_command(
        {"execution": {"command": ["python", "-u"]}}, module="pkg.wrapper", label="X", environment="x"
    ) == ("python", "-u", "-m", "pkg.wrapper")
    with pytest.raises(ConfigurationError, match="X requires execution.command .* isolated x==1 environment"):
        python_wrapper_command({"execution": {"command": " "}}, module="pkg.wrapper", label="X", environment="x==1")


def test_wrapper_check_reports_healthy_package_and_forwards_arguments(tracker: _FakeTracker) -> None:
    result = run_wrapper_check(
        tracker,
        _config(FAKE_VERSION="1.0"),
        label="Fake",
        details={"required_fake_version": "1.0"},
        check_arguments=lambda config: ("--device", "cpu"),
        version_key="fake_version",
        expected_version="1.0",
    )
    assert result["available"] is True
    assert result["problems"] == []
    assert result["tracker"] == "fake"
    assert result["required_fake_version"] == "1.0"
    assert result["package_check"] == {"arguments": ["--check", "--device", "cpu"], "fake_version": "1.0"}


def test_wrapper_check_reports_version_mismatch_before_exit_status(tracker: _FakeTracker) -> None:
    result = run_wrapper_check(
        tracker,
        _config(FAKE_VERSION="0.9", FAKE_EXIT="2"),
        label="Fake",
        version_key="fake_version",
        expected_version="1.0",
    )
    assert result["available"] is False
    assert result["problems"] == ["Fake version mismatch: expected 1.0, found 0.9."]


def test_wrapper_check_reports_wrapper_error_on_nonzero_exit(tracker: _FakeTracker) -> None:
    result = run_wrapper_check(tracker, _config(FAKE_ERROR="no model", FAKE_EXIT="2"), label="Fake")
    assert result["available"] is False
    assert result["problems"] == ["Fake wrapper check failed: no model"]


def test_wrapper_check_turns_argument_configuration_errors_into_problems(tracker: _FakeTracker) -> None:
    def invalid(config: Mapping[str, Any]) -> Sequence[str]:
        raise ConfigurationError("Fake device must be cpu.")

    result = run_wrapper_check(tracker, _config(), label="Fake", check_arguments=invalid)
    assert result["available"] is False
    assert result["problems"] == ["Fake device must be cpu."]
    assert "package_check" not in result


def test_wrapper_check_stops_when_executable_is_missing(tracker: _FakeTracker) -> None:
    config = {"execution": {"backend": "local", "command": "/definitely/missing/python"}}
    result = run_wrapper_check(tracker, config, label="Fake", details={"extra": 1})
    assert result["available"] is False
    assert result["extra"] == 1
    assert "package_check" not in result


def test_wrapper_check_skips_package_check_for_container_backend(tracker: _FakeTracker, monkeypatch) -> None:
    monkeypatch.setattr(
        ContainerExecutionBackend,
        "check",
        lambda self, command: {"backend": "container", "available": True, "problems": []},
    )
    config = {"execution": {"backend": "container", "image": "fake:1", "command": "python"}}
    result = run_wrapper_check(tracker, config, label="Fake")
    assert result["available"] is True
    assert result["package_check"] == "not_run_for_container_backend"


@pytest.mark.parametrize("value", [True, "high", None, float("nan"), float("inf")])
def test_numeric_parameter_helpers_reject_non_numbers(value: object) -> None:
    with pytest.raises(ConfigurationError, match="Tracker step must be a positive number"):
        positive_float({"step": value}, "step", label="Tracker")
    with pytest.raises(ConfigurationError, match="Tracker step must be a non-negative number"):
        nonnegative_float({"step": value}, "step", label="Tracker")


def test_numeric_parameter_helper_bounds() -> None:
    assert positive_float({"step": "0.5"}, "step", label="T") == 0.5
    assert nonnegative_float({"step": 0}, "step", label="T") == 0.0
    assert positive_int({"n": 3}, "n", label="T") == 3
    with pytest.raises(ConfigurationError, match="positive number"):
        positive_float({"step": 0}, "step", label="T")
    with pytest.raises(ConfigurationError, match="non-negative number"):
        nonnegative_float({"step": -1}, "step", label="T")
    with pytest.raises(ConfigurationError, match="must be a positive number"):
        positive_float({}, "step", label="T")
    for value in (3.0, True, 0):
        with pytest.raises(ConfigurationError, match="T n must be a positive integer"):
            positive_int({"n": value}, "n", label="T")
