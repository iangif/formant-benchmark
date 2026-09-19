"""Persistent wrapper for the upstream MLSpeech FormantsTracker inference model."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from formant_benchmark.tracker_wrappers.streaming import serve_requests

MODEL_SAMPLE_RATE = 16000
EXPECTED_TORCH_VERSION = "1.13.1"
EXPECTED_TORCHAUDIO_VERSION = "0.13.1"
DEFAULT_CHECKPOINT = "saved_ckpts/196_1125.ckpt"
_ALLOWED_DEVICES = {"auto", "cpu", "cuda"}

# These values reproduce the released conf/config.yaml architecture and inference
# preprocessing. They are intentionally not exposed as benchmark tuning parameters:
# the official checkpoint was trained for this model definition.
_MODEL_HPARAMS = {
    "sample_rate": MODEL_SAMPLE_RATE,
    "normalize": True,
    "n_fft": 512,
    "emph": 0.97,
    "gaussian_kernel_size": 7,
    "gaussian_kernel_sigma": 1,
    "f1_blocks": 2,
    "f2_blocks": 2,
    "f3_blocks": 2,
    "f4_blocks": 2,
    "f4": False,
    "bias1d": False,
    "bias": True,
    "dropout": 0.2,
}

_RUNTIME: _Runtime | None = None
_RUNTIME_KEY: tuple[str, str] | None = None


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stream", action="store_true", help="Serve JSON-lines requests on stdin/stdout.")
    mode.add_argument("--check", action="store_true", help="Check the upstream source and inference environment.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", choices=sorted(_ALLOWED_DEVICES))
    args = parser.parse_args()
    if args.check:
        return _check(args.checkpoint, args.device)
    return serve_requests(_predict)


def _predict(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    audio_path = request.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError("FormantsTracker requires a non-empty audio_path.")
    parameters = request.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("FormantsTracker request parameters must be a mapping.")
    checkpoint = parameters.get("checkpoint", DEFAULT_CHECKPOINT)
    device = parameters.get("device", "auto")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("FormantsTracker checkpoint must be a non-empty path string.")
    if device not in _ALLOWED_DEVICES:
        raise ValueError("FormantsTracker device must be one of: auto, cpu, cuda.")
    duration_s = float(request["duration_s"])
    if duration_s <= 0:
        raise ValueError("FormantsTracker request duration_s must be greater than zero.")
    runtime = _get_runtime(checkpoint, str(device))
    return runtime.predict(Path(audio_path), duration_s=duration_s)


class _Runtime:
    """One loaded checkpoint reused across all compatible requests in a worker."""

    def __init__(self, checkpoint: Path, requested_device: str) -> None:
        api = _load_api()
        self.torch = api["torch"]
        self.taudio = api["torchaudio"]
        self.taudio_functional = api["torchaudio_functional"]
        self.functional = api["functional"]
        self.dataloader = api["dataloader"]
        self.utils = api["utils"]
        self.device = _resolve_device(self.torch, requested_device)
        self.hp = SimpleNamespace(**_MODEL_HPARAMS, device=self.device, ckpt=str(checkpoint))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FormantsTracker checkpoint does not exist: {checkpoint}")

        model = api["FormantTracker"](self.hp)
        state = self.torch.load(str(checkpoint), map_location=self.device)
        model.load_state_dict(state)
        self.model = model.to(self.device)
        self.model.eval()
        self.n_bins = (self.hp.n_fft // 2) + 1
        # Preserve the released solver's frequency conversion exactly.
        self.bin_resolution = (self.hp.sample_rate / 2) / self.n_bins
        self.kernel = self.utils.get_smoothing_kernel(
            self.hp.gaussian_kernel_size,
            self.hp.gaussian_kernel_sigma,
        ).to(self.device)

    def predict(self, audio_path: Path, *, duration_s: float) -> list[dict[str, Any]]:
        """Run one WAV through released preprocessing/model/smoothing semantics."""
        if not audio_path.is_file():
            raise FileNotFoundError(f"FormantsTracker audio file does not exist: {audio_path}")
        with _model_audio(audio_path, self.taudio, self.taudio_functional) as model_audio:
            spect = self.dataloader.extract_features(str(model_audio), self.hp)
        with self.torch.no_grad():
            out, _ = self.model(spect.unsqueeze(0).to(self.device))
            smoothed = self.functional.conv2d(
                out.view(-1, 1, out.shape[2], out.shape[3]),
                self.kernel,
                padding=(self.hp.gaussian_kernel_size - 1) // 2,
            ).view(-1, 3, out.shape[2], out.shape[3])
            predicted = (
                self.torch.max(smoothed, dim=3)[1] * self.bin_resolution
            ) + self.bin_resolution / 2
            predicted = predicted[0].detach().cpu()

        rows: list[dict[str, Any]] = []
        frame_count = int(predicted.shape[1])
        for index in range(frame_count):
            time_s = index / 100.0
            # Centered STFT/resampling rounding can create a final nominal frame a
            # few samples beyond the source duration. The benchmark never stores
            # predictions outside the tracking input's coordinate support.
            if time_s > duration_s + 1e-6:
                break
            rows.append(
                {
                    "time_s": time_s,
                    "F1": float(int(predicted[0][index])),
                    "F2": float(int(predicted[1][index])),
                    "F3": float(int(predicted[2][index])),
                }
            )
        return rows


def _get_runtime(checkpoint: str, device: str) -> _Runtime:
    global _RUNTIME, _RUNTIME_KEY
    checkpoint_path = _resolve_checkpoint(checkpoint)
    key = (str(checkpoint_path), device)
    if _RUNTIME is None or _RUNTIME_KEY != key:
        _RUNTIME = _Runtime(checkpoint_path, device)
        _RUNTIME_KEY = key
    return _RUNTIME


@contextmanager
def _model_audio(audio_path: Path, taudio: Any, taudio_functional: Any) -> Iterator[Path]:
    """Yield 16 kHz audio, resampling to a temporary WAV without touching source data."""
    info = taudio.info(str(audio_path))
    source_rate = int(info.sample_rate)
    if source_rate == MODEL_SAMPLE_RATE:
        yield audio_path
        return
    if source_rate <= 0:
        raise ValueError(f"Invalid input sample rate for FormantsTracker: {source_rate}")

    waveform, loaded_rate = taudio.load(str(audio_path))
    loaded_rate = int(loaded_rate)
    if loaded_rate != source_rate:
        source_rate = loaded_rate
    resampled = taudio_functional.resample(waveform, source_rate, MODEL_SAMPLE_RATE)
    descriptor, temporary_name = tempfile.mkstemp(prefix="formants-tracker-", suffix=".wav")
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        taudio.save(str(temporary_path), resampled, MODEL_SAMPLE_RATE)
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_api() -> dict[str, Any]:
    """Import dependencies only inside the isolated tracker environment."""
    torch = importlib.import_module("torch")
    torchaudio = importlib.import_module("torchaudio")
    return {
        "torch": torch,
        "torchaudio": torchaudio,
        "torchaudio_functional": importlib.import_module("torchaudio.functional"),
        "functional": importlib.import_module("torch.nn.functional"),
        "dataloader": importlib.import_module("dataloader"),
        "utils": importlib.import_module("utils"),
        "FormantTracker": getattr(importlib.import_module("model"), "FormantTracker"),
    }


def _resolve_checkpoint(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FormantsTracker device='cuda' was requested but CUDA is unavailable.")
    return torch.device(requested)


def _check(checkpoint: str, device: str) -> int:
    details: dict[str, Any] = {
        "model_sample_rate_hz": MODEL_SAMPLE_RATE,
        "checkpoint": str(_resolve_checkpoint(checkpoint)),
        "requested_device": device,
    }
    try:
        api = _load_api()
        torch = api["torch"]
        torchaudio = api["torchaudio"]
        details["torch_version"] = str(torch.__version__)
        details["torchaudio_version"] = str(torchaudio.__version__)
        if _base_version(torch.__version__) != EXPECTED_TORCH_VERSION:
            raise RuntimeError(
                f"FormantsTracker requires torch=={EXPECTED_TORCH_VERSION}; found {torch.__version__}."
            )
        if _base_version(torchaudio.__version__) != EXPECTED_TORCHAUDIO_VERSION:
            raise RuntimeError(
                "FormantsTracker requires torchaudio=="
                f"{EXPECTED_TORCHAUDIO_VERSION}; found {torchaudio.__version__}."
            )
        checkpoint_path = _resolve_checkpoint(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"FormantsTracker checkpoint does not exist: {checkpoint_path}")
        resolved_device = _resolve_device(torch, device)
        details["resolved_device"] = str(resolved_device)
        details["cuda_available"] = bool(torch.cuda.is_available())
        details["formants_tracker_git_commit"] = _git_commit(Path.cwd())
        details["available"] = True
        print(json.dumps(details, sort_keys=True))
        return 0
    except Exception as exc:
        details["available"] = False
        details["error"] = str(exc)
        print(json.dumps(details, sort_keys=True))
        return 1


def _base_version(value: Any) -> str:
    return str(value).split("+", 1)[0]


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


if __name__ == "__main__":
    raise SystemExit(main())
