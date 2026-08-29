"""Small VTR/TIMIT source fixtures used by adapter tests."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np


def write_fb(path: Path, values_khz: np.ndarray, *, sample_period: int = 10000) -> None:
    """Write a minimal big-endian HTK `.fb` file with eight float components."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values_khz, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("VTR fixture values must have shape (n_frames, 8)")
    header = struct.pack(">IIHH", values.shape[0], sample_period, 32, 9)
    path.write_bytes(header + values.astype(">f4").tobytes())


def write_riff_wav(path: Path, *, duration_s: float = 0.05, sample_rate: int = 16000) -> None:
    """Write a tiny PCM RIFF waveform matching TIMIT sample coordinates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = round(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)


def write_sphere_stub(path: Path, *, sample_count: int, sample_rate: int = 16000) -> None:
    """Write a header-only NIST SPHERE stub sufficient for metadata parsing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header_size = 1024
    body = (
        "NIST_1A\n"
        f"{header_size:7d}\n"
        "channel_count -i 1\n"
        f"sample_count -i {sample_count}\n"
        f"sample_rate -i {sample_rate}\n"
        "sample_n_bytes -i 2\n"
        "end_head\n"
    ).encode("ascii")
    if len(body) > header_size:
        raise ValueError("SPHERE fixture header exceeded fixed header size")
    path.write_bytes(body + b" " * (header_size - len(body)))


def create_vtr_layout(root: Path, *, sphere_audio: bool = False) -> tuple[Path, Path]:
    """Create one train and one test VTR utterance with disjoint speakers."""
    vtr_root = root / "VTRFormants"
    audio_root = root / "TIMIT_fixed"
    values = np.array(
        [
            [0.50, 1.50, 2.50, 3.50, 0.10, 0.20, 0.30, 0.40],
            [0.51, 1.51, 2.51, 3.51, 0.10, 0.20, 0.30, 0.40],
            [0.52, 1.52, 2.52, 3.52, 0.10, 0.20, 0.30, 0.40],
            [0.53, 1.53, 2.53, 3.53, 0.10, 0.20, 0.30, 0.40],
            [0.54, 1.54, 2.54, 3.54, 0.10, 0.20, 0.30, 0.40],
        ],
        dtype=np.float32,
    )

    cases = (
        ("Train", "TRAIN", "dr1", "fcjf0", "sx1"),
        ("Test", "TEST", "dr2", "mtas1", "si2"),
    )
    for source_split, audio_split, dialect, speaker, sentence in cases:
        base = vtr_root / source_split / dialect / speaker / sentence
        write_fb(base.with_suffix(".fb"), values)
        base.with_suffix(".phn").write_text(
            "0 160 h#\n160 480 iy\n480 800 t\n",
            encoding="ascii",
        )
        base.with_suffix(".wrd").write_text("160 800 eat\n", encoding="ascii")
        wav_path = audio_root / audio_split / dialect.upper() / speaker.upper() / f"{sentence.upper()}.WAV"
        if sphere_audio:
            write_sphere_stub(wav_path, sample_count=800)
        else:
            write_riff_wav(wav_path)

    return vtr_root, audio_root
