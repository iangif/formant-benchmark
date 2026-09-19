"""FormantsTracker adapter and dependency-isolated wrapper tests."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from formant_benchmark.data.models import Formant, TrackingInputMode
from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.tracker_wrappers import formants_tracker as wrapper
from formant_benchmark.trackers.formants_tracker import (
    FORMANTS_TRACKER_DEFAULT_CHECKPOINT,
    FORMANTS_TRACKER_MODEL_SAMPLE_RATE,
    FormantsTrackerTracker,
)
from tests.integration_helpers import configured_real_tracker


def test_formants_tracker_capabilities_and_defaults() -> None:
    tracker = FormantsTrackerTracker()
    assert tracker.capabilities.formants == frozenset({Formant.F1, Formant.F2, Formant.F3})
    assert tracker.capabilities.input_modes == frozenset(
        {TrackingInputMode.FULL_ITEM, TrackingInputMode.CROPPED_INTERVALS}
    )
    assert tracker.capabilities.interval_types == frozenset({"phone", "vowel", "word"})
    assert tracker.default_configuration["parameters"] == {
        "checkpoint": FORMANTS_TRACKER_DEFAULT_CHECKPOINT,
        "device": "auto",
    }


def test_formants_tracker_wrapper_command_uses_configured_python() -> None:
    tracker = FormantsTrackerTracker()
    command = tracker.wrapper_command({"execution": {"command": ["python-ft", "-u"]}})
    assert command == (
        "python-ft",
        "-u",
        "-m",
        "formant_benchmark.tracker_wrappers.formants_tracker",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"unknown": 1}, "Unsupported FormantsTracker parameter"),
        ({"checkpoint": ""}, "checkpoint"),
        ({"device": "mps"}, "device"),
    ],
)
def test_formants_tracker_rejects_invalid_parameters(
    changes: dict[str, object], message: str
) -> None:
    tracker = FormantsTrackerTracker()
    parameters = dict(tracker.default_configuration["parameters"])
    parameters.update(changes)
    with pytest.raises(ConfigurationError, match=message):
        tracker.validate_parameters(parameters)


def test_wrapper_reuses_runtime_for_same_checkpoint_and_device(monkeypatch, tmp_path: Path) -> None:
    created: list[tuple[Path, str]] = []

    class FakeRuntime:
        def __init__(self, checkpoint: Path, device: str) -> None:
            created.append((checkpoint, device))

        def predict(self, audio_path: Path, *, duration_s: float):
            return [{"time_s": 0.0, "F1": 500.0, "F2": 1500.0, "F3": 2500.0}]

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wrapper, "_Runtime", FakeRuntime)
    monkeypatch.setattr(wrapper, "_RUNTIME", None)
    monkeypatch.setattr(wrapper, "_RUNTIME_KEY", None)

    first = wrapper._get_runtime("model.ckpt", "cpu")
    second = wrapper._get_runtime("model.ckpt", "cpu")
    assert first is second
    assert created == [(checkpoint.resolve(), "cpu")]

    wrapper._get_runtime("model.ckpt", "auto")
    assert created == [
        (checkpoint.resolve(), "cpu"),
        (checkpoint.resolve(), "auto"),
    ]


def test_predict_forwards_checkpoint_device_and_duration(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class FakeRuntime:
        def predict(self, audio_path: Path, *, duration_s: float):
            observed["audio_path"] = audio_path
            observed["duration_s"] = duration_s
            return [{"time_s": 0.0, "F1": 500.0, "F2": 1500.0, "F3": 2500.0}]

    def get_runtime(checkpoint: str, device: str):
        observed["checkpoint"] = checkpoint
        observed["device"] = device
        return FakeRuntime()

    monkeypatch.setattr(wrapper, "_get_runtime", get_runtime)
    rows = wrapper._predict(
        {
            "audio_path": str(tmp_path / "input.wav"),
            "duration_s": 0.25,
            "parameters": {"checkpoint": "custom.ckpt", "device": "cpu"},
        }
    )
    assert observed == {
        "checkpoint": "custom.ckpt",
        "device": "cpu",
        "audio_path": tmp_path / "input.wav",
        "duration_s": 0.25,
    }
    assert rows[0] == {"time_s": 0.0, "F1": 500.0, "F2": 1500.0, "F3": 2500.0}


def test_model_audio_resamples_non_16khz_to_temporary_wav(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    observed: dict[str, object] = {}

    class FakeTorchaudio:
        @staticmethod
        def info(path: str):
            assert path == str(source)
            return SimpleNamespace(sample_rate=8000)

        @staticmethod
        def load(path: str):
            assert path == str(source)
            return "waveform", 8000

        @staticmethod
        def save(path: str, waveform, sample_rate: int) -> None:
            observed["saved_path"] = path
            observed["waveform"] = waveform
            observed["sample_rate"] = sample_rate
            Path(path).write_bytes(b"resampled")

    class FakeFunctional:
        @staticmethod
        def resample(waveform, source_rate: int, target_rate: int):
            observed["resample"] = (waveform, source_rate, target_rate)
            return "resampled-waveform"

    with wrapper._model_audio(source, FakeTorchaudio, FakeFunctional) as model_audio:
        assert model_audio != source
        assert model_audio.is_file()
        temporary = model_audio

    assert not temporary.exists()
    assert source.read_bytes() == b"source"
    assert observed["resample"] == ("waveform", 8000, FORMANTS_TRACKER_MODEL_SAMPLE_RATE)
    assert observed["waveform"] == "resampled-waveform"
    assert observed["sample_rate"] == FORMANTS_TRACKER_MODEL_SAMPLE_RATE


def test_model_audio_leaves_16khz_input_untouched(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")

    class FakeTorchaudio:
        @staticmethod
        def info(path: str):
            return SimpleNamespace(sample_rate=FORMANTS_TRACKER_MODEL_SAMPLE_RATE)

    with wrapper._model_audio(source, FakeTorchaudio, object()) as model_audio:
        assert model_audio == source


def test_wrapper_check_validates_pinned_stack_and_checkpoint(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        __version__ = "1.13.1+cpu"
        cuda = FakeCuda()

        @staticmethod
        def device(value: str) -> str:
            return value

    fake_taudio = SimpleNamespace(__version__="0.13.1+cpu")
    monkeypatch.setattr(
        wrapper,
        "_load_api",
        lambda: {
            "torch": FakeTorch,
            "torchaudio": fake_taudio,
            "torchaudio_functional": object(),
            "functional": object(),
            "dataloader": object(),
            "utils": object(),
            "FormantTracker": object(),
        },
    )
    monkeypatch.setattr(wrapper, "_git_commit", lambda root: "abc123")
    monkeypatch.chdir(tmp_path)

    assert wrapper._check("model.ckpt", "auto") == 0
    details = json.loads(capsys.readouterr().out)
    assert details["available"] is True
    assert details["resolved_device"] == "cpu"
    assert details["formants_tracker_git_commit"] == "abc123"


def test_real_formants_tracker_environment_when_installed(tmp_path: Path) -> None:
    """Run real FormantsTracker automatically when its configured optional install exists."""
    tracker = FormantsTrackerTracker()
    effective, backend, command = configured_real_tracker(tracker)

    # Deliberately use 8 kHz input: the wrapper should transparently resample it
    # to the model's required 16 kHz without changing the source file.
    audio = tmp_path / "vowel-8k.wav"
    _write_voiced_wav(audio, sample_rate=8000, duration_s=0.2)
    request = {
        "type": "request",
        "protocol_version": "2",
        "tracker": "formants_tracker",
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
        result = worker.request(request, timeout_s=120)
    finally:
        worker.close()

    assert result.returncode == 0, result.stderr
    assert result.response is not None
    assert result.response["type"] == "result", result.response
    rows = result.response["rows"]
    assert rows
    assert set(rows[0]) == {"time_s", "F1", "F2", "F3"}
    assert rows[1]["time_s"] == pytest.approx(0.01)


def _write_voiced_wav(path: Path, *, sample_rate: int, duration_s: float) -> None:
    import math
    import struct

    frames = bytearray()
    for index in range(round(sample_rate * duration_s)):
        t = index / sample_rate
        value = sum(
            math.sin(2 * math.pi * frequency * t)
            for frequency in (120, 500, 1500, 2500)
        ) / 4
        frames.extend(struct.pack("<h", int(12000 * value)))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(bytes(frames))
