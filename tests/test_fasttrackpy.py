"""FastTrackPy adapter and dependency-isolated wrapper tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from formant_benchmark.config.resolution import resolve_parameters
from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.tracker_wrappers import fasttrackpy as wrapper
from formant_benchmark.trackers.fasttrackpy import FASTTRACKPY_VERSION, FastTrackPyTracker


def test_fasttrackpy_capabilities_match_phase_7_scope() -> None:
    tracker = FastTrackPyTracker()
    assert tracker.version == "0.6.1"
    assert tracker.capabilities.formants == frozenset(Formant)
    assert tracker.capabilities.input_modes == frozenset(
        {TrackingInputMode.FULL_ITEM, TrackingInputMode.CROPPED_INTERVALS}
    )
    assert TrackingInputMode.FULL_ITEM_WITH_INTERVALS not in tracker.capabilities.input_modes


def test_fasttrackpy_defaults_and_metadata_overrides_validate() -> None:
    tracker = FastTrackPyTracker()
    parameters = resolve_parameters(
        {"gender": "female"},
        tracker.default_configuration,
        {
            "overrides": [
                {
                    "where": {"gender": "female"},
                    "parameters": {"min_max_formant": 4500.0, "max_max_formant": 7500.0},
                }
            ]
        },
        {"parameters": {"nstep": 25}},
    )
    tracker.validate_parameters(parameters)
    assert parameters["min_max_formant"] == 4500.0
    assert parameters["max_max_formant"] == 7500.0
    assert parameters["nstep"] == 25
    assert parameters["smoother_method"] == "dct_smooth_regression"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"unknown": 1}, "Unsupported FastTrackPy parameter"),
        ({"max_max_formant": 3000}, "greater than min_max_formant"),
        ({"n_formants": 5}, "at most 4"),
        ({"smoother_method": "mystery"}, "smoother_method"),
        ({"heuristics": ["mystery"]}, "heuristics"),
    ],
)
def test_fasttrackpy_rejects_invalid_parameters(changes: dict[str, object], message: str) -> None:
    tracker = FastTrackPyTracker()
    parameters = dict(tracker.default_configuration["parameters"])
    parameters.update(changes)
    with pytest.raises(ConfigurationError, match=message):
        tracker.validate_parameters(parameters)


def test_wrapper_emits_smoothed_winner_and_forwards_parameters(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class Factory:
        def __init__(self, **kwargs):
            observed.setdefault("factories", []).append(kwargs)

    winner = SimpleNamespace(
        time_domain=[0.0, 0.01, 0.02],
        smoothed_formants=[
            [500.0, 510.0, 520.0],
            [1500.0, 1510.0, 1520.0],
            [2500.0, 2510.0, 2520.0],
            [3500.0, 3510.0, 3520.0],
        ],
        formants=[[1.0, 1.0, 1.0]] * 4,
    )

    def process_audio_file(path, **kwargs):
        observed["path"] = path
        observed["kwargs"] = kwargs
        return SimpleNamespace(winner=winner)

    monkeypatch.setattr(
        wrapper,
        "_load_api",
        lambda: {
            "process_audio_file": process_audio_file,
            "Smoother": Factory,
            "Loss": Factory,
            "Agg": Factory,
        },
    )
    monkeypatch.setattr(wrapper, "_resolve_heuristics", lambda names: ["heuristic"] if names else [])

    parameters = dict(FastTrackPyTracker.default_configuration["parameters"])
    parameters["heuristics"] = ["f1_max"]
    rows = wrapper._predict({"audio_path": "vowel.wav", "parameters": parameters})

    assert observed["path"] == "vowel.wav"
    assert rows[1] == {
        "time_s": 0.01,
        "F1": 510.0,
        "F2": 1510.0,
        "F3": 2510.0,
        "F4": 3510.0,
    }
    kwargs = observed["kwargs"]
    assert kwargs["min_max_formant"] == 4000.0
    assert kwargs["nstep"] == 20
    assert kwargs["heuristics"] == ["heuristic"]
    assert "formants" not in rows[0]  # raw winner.formants is intentionally ignored


def test_wrapper_supports_fewer_than_four_smoothed_formants(monkeypatch) -> None:
    winner = SimpleNamespace(
        time_domain=[0.0, 0.01],
        smoothed_formants=[[500.0, 510.0], [1500.0, 1510.0], [2500.0, 2510.0]],
    )
    monkeypatch.setattr(
        wrapper,
        "_load_api",
        lambda: {
            "process_audio_file": lambda *args, **kwargs: SimpleNamespace(winner=winner),
            "Smoother": lambda **kwargs: None,
            "Loss": lambda **kwargs: None,
            "Agg": lambda **kwargs: None,
        },
    )
    rows = wrapper._predict(
        {
            "audio_path": "vowel.wav",
            "parameters": FastTrackPyTracker.default_configuration["parameters"],
        }
    )
    assert rows[0]["F3"] == 2500.0
    assert rows[0]["F4"] is None


def test_persistent_wrapper_check_and_request_with_fake_fasttrackpy(tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_site"
    package = fake_root / "fasttrackpy"
    processors = package / "processors"
    dist_info = fake_root / f"fasttrackpy-{FASTTRACKPY_VERSION}.dist-info"
    processors.mkdir(parents=True)
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: fasttrackpy\nVersion: {FASTTRACKPY_VERSION}\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        """
class Smoother:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class Loss:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class Agg:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class Winner:
    time_domain = [0.0, 0.01]
    smoothed_formants = [
        [500.0, 510.0],
        [1500.0, 1510.0],
        [2500.0, 2510.0],
        [3500.0, 3510.0],
    ]

class Candidates:
    winner = Winner()

def process_audio_file(path, **kwargs):
    return Candidates()
""",
        encoding="utf-8",
    )
    (processors / "__init__.py").write_text(
        """
class Heuristics:
    F1_Max = "F1_Max"
    F4_Min = "F4_Min"
    B2_Max = "B2_Max"
    B3_Max = "B3_Max"
    Rhotic = "Rhotic"
    F3_F4_Sep = "F3_F4_Sep"

heuristic = Heuristics()
""",
        encoding="utf-8",
    )

    src_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(fake_root), str(src_root)])
    command = [sys.executable, "-m", "formant_benchmark.tracker_wrappers.fasttrackpy"]

    checked = subprocess.run(
        [*command, "--check"], env=environment, capture_output=True, text=True, check=False
    )
    assert checked.returncode == 0
    assert json.loads(checked.stdout)["fasttrackpy_version"] == FASTTRACKPY_VERSION

    request = {
        "type": "request",
        "protocol_version": "2",
        "tracker": "fasttrackpy",
        "item_id": "token-1",
        "input_unit_id": "token-1",
        "audio_path": "vowel.wav",
        "duration_s": 0.1,
        "parameters": FastTrackPyTracker.default_configuration["parameters"],
        "metadata": {},
        "intervals": [],
    }
    streamed = subprocess.run(
        [*command, "--stream"],
        env=environment,
        input=json.dumps(request) + "\n" + json.dumps({"type": "shutdown"}) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert streamed.returncode == 0
    response = json.loads(streamed.stdout.splitlines()[0])
    assert response["type"] == "result"
    assert response["rows"][0]["F4"] == 3500.0


def test_real_fasttrackpy_environment_when_configured(tmp_path: Path) -> None:
    """Optional smoke test: set FASTTRACKPY_PYTHON to exercise the real 0.6.1 install."""
    python = os.environ.get("FASTTRACKPY_PYTHON")
    if not python:
        pytest.skip("Set FASTTRACKPY_PYTHON to the isolated fasttrackpy==0.6.1 Python executable.")

    tracker = FastTrackPyTracker()
    src_root = Path(__file__).resolve().parents[1] / "src"
    config = {
        "execution": {
            "backend": "local",
            "command": python,
            "environment": {"PYTHONPATH": str(src_root)},
        }
    }
    effective = {**tracker.default_configuration, **config}
    check = tracker.check_environment(effective)
    assert check["available"] is True, check

    audio = tmp_path / "vowel.wav"
    _write_voiced_wav(audio)
    request = {
        "type": "request",
        "protocol_version": "2",
        "tracker": "fasttrackpy",
        "item_id": "real-smoke",
        "input_unit_id": "real-smoke",
        "audio_path": str(audio),
        "duration_s": 0.2,
        "parameters": dict(tracker.default_configuration["parameters"]),
        "metadata": {},
        "intervals": [],
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(src_root)
    command = [python, "-m", "formant_benchmark.tracker_wrappers.fasttrackpy", "--stream"]
    completed = subprocess.run(
        command,
        env=environment,
        input=json.dumps(request) + "\n" + json.dumps({"type": "shutdown"}) + "\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout.splitlines()[0])
    assert response["type"] == "result", response
    assert response["rows"]
    assert any(row.get("F1") is not None for row in response["rows"])


def _write_voiced_wav(path: Path, sample_rate: int = 16000, duration_s: float = 0.2) -> None:
    import math
    import struct

    frames = bytearray()
    for index in range(round(sample_rate * duration_s)):
        t = index / sample_rate
        value = sum(math.sin(2 * math.pi * frequency * t) for frequency in (120, 500, 1500, 2500)) / 4
        frames.extend(struct.pack("<h", int(12000 * value)))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(bytes(frames))
