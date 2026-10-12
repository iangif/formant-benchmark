"""Praat Burg adapter and dependency-isolated Parselmouth wrapper tests."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from formant_benchmark.config.resolution import resolve_parameters
from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.tracker_wrappers import praat_burg as wrapper
from formant_benchmark.trackers import TRACKER_REGISTRY, register_builtin_trackers
from formant_benchmark.trackers.praat_burg import PARSELMOUTH_VERSION, PraatBurgTracker
from tests.integration_helpers import configured_real_tracker


def test_praat_burg_capabilities_and_registration() -> None:
    tracker = PraatBurgTracker()
    assert tracker.version == PARSELMOUTH_VERSION
    assert tracker.capabilities.formants == frozenset(Formant)
    assert tracker.capabilities.input_modes == frozenset(
        {TrackingInputMode.FULL_ITEM, TrackingInputMode.CROPPED_INTERVALS}
    )
    register_builtin_trackers()
    assert TRACKER_REGISTRY.get("praat_burg") is PraatBurgTracker


def test_praat_burg_wrapper_command_uses_configured_python() -> None:
    command = PraatBurgTracker().wrapper_command({"execution": {"command": "/env/bin/python"}})
    assert tuple(command) == ("/env/bin/python", "-m", "formant_benchmark.tracker_wrappers.praat_burg")
    with pytest.raises(ConfigurationError, match="execution.command"):
        PraatBurgTracker().wrapper_command({"execution": {}})


def test_praat_burg_defaults_and_metadata_overrides_validate() -> None:
    tracker = PraatBurgTracker()
    parameters = resolve_parameters(
        {"gender": "male"},
        tracker.default_configuration,
        {"overrides": [{"where": {"gender": "male"}, "parameters": {"maximum_formant": 5000.0}}]},
        {"parameters": {"time_step": 0.005}},
    )
    tracker.validate_parameters(parameters)
    assert parameters == {
        "time_step": 0.005,
        "max_number_of_formants": 5.0,
        "maximum_formant": 5000.0,
        "window_length": 0.025,
        "pre_emphasis_from": 50.0,
    }
    tracker.validate_parameters(tracker.default_configuration["parameters"])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"unknown": 1}, "Unsupported Praat Burg parameter"),
        ({"time_step": 0}, "time_step"),
        ({"max_number_of_formants": 5.25}, "multiple of 0.5"),
        ({"max_number_of_formants": -1}, "max_number_of_formants"),
        ({"maximum_formant": "high"}, "maximum_formant"),
        ({"window_length": True}, "window_length"),
        ({"pre_emphasis_from": float("nan")}, "pre_emphasis_from"),
    ],
)
def test_praat_burg_rejects_invalid_parameters(changes: dict[str, object], message: str) -> None:
    tracker = PraatBurgTracker()
    parameters = dict(tracker.default_configuration["parameters"])
    parameters.update(changes)
    with pytest.raises(ConfigurationError, match=message):
        tracker.validate_parameters(parameters)


class _FakeFormant:
    """Frame values for formant numbers 1..5; NaN marks a formant Praat did not find."""

    def __init__(self, frames: dict[float, list[float]]) -> None:
        self._frames = frames

    def xs(self) -> list[float]:
        return list(self._frames)

    def get_value_at_time(self, number: int, time: float) -> float:
        return self._frames[time][number - 1]


class _FakeSound:
    def __init__(self, path: str, observed: dict[str, object], n_channels: int = 1) -> None:
        observed["path"] = path
        self._observed = observed
        self.n_channels = n_channels

    def convert_to_mono(self) -> "_FakeSound":
        self._observed["mono"] = True
        return _FakeSound(str(self._observed["path"]), self._observed)

    def to_formant_burg(self, **kwargs: object) -> _FakeFormant:
        self._observed["kwargs"] = kwargs
        nan = float("nan")
        return _FakeFormant(
            {
                0.0125: [500.0, 1500.0, 2500.0, 3500.0, 4500.0],
                0.01875: [510.0, 1510.0, 2510.0, nan, nan],
            }
        )


def _fake_parselmouth(observed: dict[str, object], n_channels: int = 1) -> object:
    class Module:
        @staticmethod
        def Sound(path: str) -> _FakeSound:
            return _FakeSound(path, observed, n_channels)

    return Module


def test_wrapper_emits_f1_to_f4_per_frame_and_forwards_parameters(monkeypatch) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(wrapper, "_load_parselmouth", lambda: _fake_parselmouth(observed))

    rows = wrapper._predict(
        {"audio_path": "vowel.wav", "parameters": PraatBurgTracker.default_configuration["parameters"]}
    )

    assert observed["path"] == "vowel.wav"
    assert observed["kwargs"] == {
        "time_step": None,
        "max_number_of_formants": 5.0,
        "maximum_formant": 5500.0,
        "window_length": 0.025,
        "pre_emphasis_from": 50.0,
    }
    # F5 is dropped; formants Praat did not find become missing rather than failures.
    assert rows == [
        {"time_s": 0.0125, "F1": 500.0, "F2": 1500.0, "F3": 2500.0, "F4": 3500.0},
        {"time_s": 0.01875, "F1": 510.0, "F2": 1510.0, "F3": 2510.0, "F4": None},
    ]
    assert "mono" not in observed


def test_wrapper_mixes_multichannel_audio_to_mono(monkeypatch) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(wrapper, "_load_parselmouth", lambda: _fake_parselmouth(observed, n_channels=2))
    parameters = dict(PraatBurgTracker.default_configuration["parameters"], time_step=0.005)
    wrapper._predict({"audio_path": "stereo.wav", "parameters": parameters})
    assert observed["mono"] is True
    assert observed["kwargs"]["time_step"] == 0.005  # type: ignore[index]


def test_persistent_wrapper_check_and_request_with_fake_parselmouth(tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_site"
    dist_info = fake_root / f"praat_parselmouth-{PARSELMOUTH_VERSION}.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: praat-parselmouth\nVersion: {PARSELMOUTH_VERSION}\n",
        encoding="utf-8",
    )
    (fake_root / "parselmouth.py").write_text(
        """
PRAAT_VERSION = "6.1.38"

class Formant:
    def xs(self):
        return [0.0125, 0.01875]

    def get_value_at_time(self, number, time):
        return [500.0, 1500.0, 2500.0, 3500.0, 4500.0][number - 1] + (10.0 if time > 0.015 else 0.0)

class Sound:
    n_channels = 1

    def __init__(self, path):
        self.path = path

    def to_formant_burg(self, **kwargs):
        return Formant()
""",
        encoding="utf-8",
    )

    src_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(fake_root), str(src_root)])
    command = [sys.executable, "-m", "formant_benchmark.tracker_wrappers.praat_burg"]

    checked = subprocess.run(
        [*command, "--check"], env=environment, capture_output=True, text=True, check=False
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == {
        "parselmouth_version": PARSELMOUTH_VERSION,
        "praat_version": "6.1.38",
    }

    request = {
        "type": "request",
        "protocol_version": "2",
        "tracker": "praat_burg",
        "item_id": "token-1",
        "input_unit_id": "token-1",
        "audio_path": "vowel.wav",
        "duration_s": 0.1,
        "parameters": PraatBurgTracker.default_configuration["parameters"],
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
    assert streamed.returncode == 0, streamed.stderr
    response = json.loads(streamed.stdout.splitlines()[0])
    assert response["type"] == "result"
    assert response["rows"][1] == {"time_s": 0.01875, "F1": 510.0, "F2": 1510.0, "F3": 2510.0, "F4": 3510.0}


def test_real_praat_burg_environment_when_installed(tmp_path: Path) -> None:
    """Run real Parselmouth automatically when its configured optional install exists."""
    tracker = PraatBurgTracker()
    effective, backend, command = configured_real_tracker(tracker)

    audio = tmp_path / "vowel.wav"
    _write_voiced_wav(audio)
    request = {
        "type": "request",
        "protocol_version": "2",
        "tracker": "praat_burg",
        "item_id": "real-smoke",
        "input_unit_id": "real-smoke",
        "audio_path": str(audio),
        "duration_s": 0.2,
        "parameters": dict(effective["parameters"]),
        "metadata": {},
        "intervals": [],
    }

    worker = backend.start_worker(command, tmp_path)
    try:
        result = worker.request(request, timeout_s=60)
    finally:
        worker.close()

    assert result.returncode == 0, result.stderr
    assert result.response is not None
    assert result.response["type"] == "result", result.response
    rows = result.response["rows"]
    assert rows
    assert all(0.0 <= row["time_s"] <= 0.2 for row in rows)
    # The synthetic vowel has strong components at 500 and 1500 Hz.
    mid = rows[len(rows) // 2]
    assert mid["F1"] == pytest.approx(500.0, abs=50.0)
    assert mid["F2"] == pytest.approx(1500.0, abs=100.0)


def _write_voiced_wav(path: Path, sample_rate: int = 16000, duration_s: float = 0.2) -> None:
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
