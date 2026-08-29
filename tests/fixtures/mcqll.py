"""Tiny synthetic source fixtures matching the current formants-export contract."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pandas as pd


def write_silent_wav(path: Path, *, duration_s: float = 0.20, sample_rate: int = 8000) -> None:
    """Write a tiny PCM WAV suitable for adapter duration/path tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def mcqll_source_frames(
    *,
    corpus: str,
    batch: str,
    token_prefix: str,
    include_f4: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return token/track frames with one exported and one excluded token."""
    exported_id = f"{token_prefix}:exported"
    excluded_id = f"{token_prefix}:excluded"
    metadata = {
        "speaker": f"speaker-{token_prefix}",
        "gender": "female",
        "discourse": "story1",
        "linguistic": {
            "phone": "AE1",
            "ipa": "æ",
            "syllable": "cat",
            "word": "cat",
            "transcription": "cat",
            "previous": {"phone": "K", "ipa": "k"},
            "following": {"phone": "T", "ipa": "t"},
        },
        "intervals": {
            "phone": {
                "begin": 10.05,
                "end": 10.15,
                "corrected_begin": 10.06,
                "corrected_end": 10.14,
            },
            "syllable": {"begin": 10.0, "end": 10.2},
            "word": {"begin": 10.0, "end": 10.2},
            "clip": {"begin": 10.0, "end": 10.2},
        },
        "alignment": {"value": "ok", "comment": None},
    }
    tokens = pd.DataFrame(
        [
            {
                "token_id": exported_id,
                "corpus": corpus,
                "batch": batch,
                "file_stem": f"{token_prefix}_exported",
                "metadata": metadata,
                "export": {"status": "exported", "reason": "agreement"},
            },
            {
                "token_id": excluded_id,
                "corpus": corpus,
                "batch": batch,
                "file_stem": f"{token_prefix}_excluded",
                "metadata": metadata,
                "export": {"status": "excluded", "reason": "bad_token"},
            },
        ]
    )
    tracks = pd.DataFrame(
        {
            "token_id": [exported_id, exported_id],
            "time": [0.05, 0.15],
            "F1": [500.0, 510.0],
            "F2": [1500.0, 1510.0],
            "F3": [2500.0, 2510.0],
            "F4": [3500.0, 3510.0] if include_f4 else [None, None],
            "F1_s": [550.0, 560.0],
            "F2_s": [1550.0, 1560.0],
            "F3_s": [2550.0, 2560.0],
            "F4_s": [3550.0, 3560.0] if include_f4 else [None, None],
        }
    )
    return tokens, tracks


def create_source_layout(
    root: Path,
    *,
    corpus: str,
    batches: tuple[str, ...],
    include_f4: bool = True,
) -> tuple[Path, Path, dict[Path, pd.DataFrame]]:
    """Create manifests/audio/files and return frames keyed by fake Parquet path."""
    gold_root = root / "gold_tracks"
    audio_root = root / "source"
    parquet_frames: dict[Path, pd.DataFrame] = {}

    for index, batch in enumerate(batches, start=1):
        batch_root = gold_root / batch
        batch_root.mkdir(parents=True, exist_ok=True)
        (batch_root / "tracks.parquet").touch()
        (batch_root / "tokens.parquet").touch()
        (batch_root / "export_manifest.json").write_text(
            json.dumps({"corpus": corpus, "batch": batch, "schema": "gold_export_v5"}),
            encoding="utf-8",
        )
        token_prefix = f"t{index}"
        tokens, tracks = mcqll_source_frames(
            corpus=corpus,
            batch=batch,
            token_prefix=token_prefix,
            include_f4=include_f4,
        )
        parquet_frames[batch_root / "tokens.parquet"] = tokens
        parquet_frames[batch_root / "tracks.parquet"] = tracks
        write_silent_wav(audio_root / batch / "audio" / f"{token_prefix}_exported.wav")

    return gold_root, audio_root, parquet_frames
