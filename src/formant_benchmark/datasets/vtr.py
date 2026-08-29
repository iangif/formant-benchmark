"""Prepare the MSR-UCLA VTR-Formant database with matching TIMIT audio."""

from __future__ import annotations

import re
import struct
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from formant_benchmark.data.models import AnnotationType, DatasetManifest, Formant, IntervalOrigin, IntervalType, PreparedDataset
from formant_benchmark.data.schemas import FORMANT_COLUMNS
from formant_benchmark.datasets.base import DatasetAdapter
from formant_benchmark.exceptions import ConfigurationError, DatasetValidationError
from formant_benchmark.preparation.intervals import derive_vowel_intervals

# The VTR user manual states that successive .fb vectors are 10 ms apart. The
# bundled HTK header reports a different period, so preparation follows the
# database's documented frame semantics rather than that inconsistent header field.
_VTR_FRAME_STEP_S = 0.010
_VTR_AVAILABLE_FORMANTS = (Formant.F1, Formant.F2, Formant.F3, Formant.F4)
_VTR_COMPONENTS = 8
_HTK_HEADER_SIZE = 12
_SPHERE_MAGIC = b"NIST_1A"

_DIALECT_NAMES = {
    "dr1": "New England",
    "dr2": "Northern",
    "dr3": "North Midland",
    "dr4": "South Midland",
    "dr5": "Southern",
    "dr6": "New York City",
    "dr7": "Western",
    "dr8": "Army Brat",
}

# TIMIT vowel phones. Syllabic consonants (el/em/en/eng) are intentionally not
# included; callers can override this list in dataset configuration if desired.
DEFAULT_VOWEL_LABELS = (
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ax",
    "ax-h",
    "axr",
    "ay",
    "eh",
    "er",
    "ey",
    "ih",
    "ix",
    "iy",
    "ow",
    "oy",
    "uh",
    "uw",
    "ux",
)


class VTRConfig(BaseModel):
    """Source-specific configuration for preparing VTR with TIMIT waveforms."""

    model_config = ConfigDict(extra="allow")

    adapter: str = "vtr"
    name: str = Field(default="vtr", min_length=1)
    vtr_root: Path
    audio_root: Path
    language: str = Field(default="english", min_length=1)
    vowel_labels: list[str] = Field(default_factory=lambda: list(DEFAULT_VOWEL_LABELS))

    @field_validator("adapter")
    @classmethod
    def validate_adapter(cls, value: str) -> str:
        if value != "vtr":
            raise ValueError("VTRConfig requires adapter: vtr")
        return value

    @field_validator("vowel_labels")
    @classmethod
    def validate_vowel_labels(cls, value: list[str]) -> list[str]:
        cleaned = [str(label).strip().lower() for label in value if str(label).strip()]
        if not cleaned:
            raise ValueError("vowel_labels must contain at least one phone label")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("vowel_labels must not contain duplicates")
        return sorted(cleaned)


class VTRAdapter(DatasetAdapter):
    """Adapt VTR utterance trajectories while preserving TIMIT-native items.

    Each VTR ``.fb`` file becomes one utterance item. Source phone and word
    boundaries remain intervals within that item, and vowel intervals are derived
    from the configured TIMIT vowel inventory. VTR provides F1-F4, so all four
    formants are preserved in the prepared dataset. Evaluation later determines
    which subset of available formants should be scored.
    """

    name = "vtr"
    version = "1"

    def prepare(self, config: Mapping[str, Any]) -> PreparedDataset:
        try:
            parsed = VTRConfig.model_validate(dict(config))
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid VTR dataset configuration: {exc}") from exc

        vtr_root = parsed.vtr_root.expanduser().resolve()
        audio_root = parsed.audio_root.expanduser().resolve()
        split_roots = _resolve_split_roots(vtr_root)
        if not audio_root.is_dir():
            raise DatasetValidationError(f"VTR audio_root does not exist or is not a directory: {audio_root}")

        item_rows: list[dict[str, Any]] = []
        track_frames: list[pd.DataFrame] = []
        interval_rows: list[dict[str, Any]] = []
        split_rows: list[dict[str, str]] = []
        header_periods: set[int] = set()
        skipped_zero_length_words = 0

        for split, split_root in split_roots.items():
            fb_paths = sorted(split_root.glob("dr*/*/*.fb"))
            if not fb_paths:
                raise DatasetValidationError(f"VTR source split '{split}' contains no .fb files: {split_root}")

            for fb_path in fb_paths:
                relative = fb_path.relative_to(split_root)
                if len(relative.parts) != 3:
                    raise DatasetValidationError(f"Unexpected VTR path layout: {fb_path}")
                dialect, speaker_dir, filename = relative.parts
                sentence_id = Path(filename).stem
                item_id = f"vtr:{split}:{dialect.lower()}:{speaker_dir.lower()}:{sentence_id.lower()}"

                phn_path = fb_path.with_suffix(".phn")
                wrd_path = fb_path.with_suffix(".wrd")
                if not phn_path.is_file() or not wrd_path.is_file():
                    raise DatasetValidationError(
                        f"VTR utterance '{fb_path}' requires matching .phn and .wrd files."
                    )

                audio_path = _resolve_audio_path(
                    audio_root,
                    split=split,
                    dialect=dialect,
                    speaker_dir=speaker_dir,
                    sentence_id=sentence_id,
                )
                sample_rate, sample_count = _read_audio_info(audio_path)
                duration_s = sample_count / sample_rate

                tracks, header_period = _read_fb(fb_path, item_id=item_id)
                header_periods.add(header_period)
                if not tracks.empty and float(tracks["time_s"].iloc[-1]) > duration_s + 1e-6:
                    raise DatasetValidationError(
                        f"VTR track extends beyond matching TIMIT audio for '{item_id}': "
                        f"last track time={tracks['time_s'].iloc[-1]:.6f}s, duration={duration_s:.6f}s."
                    )
                track_frames.append(tracks)

                gender = _speaker_gender(speaker_dir)
                item_rows.append(
                    {
                        "item_id": item_id,
                        "source": "vtr",
                        "item_type": "utterance",
                        "speaker_id": speaker_dir[1:].lower(),
                        "gender": gender,
                        "age": None,
                        "language": parsed.language,
                        "dialect": dialect.lower(),
                        "audio_path": str(audio_path),
                        "duration_s": duration_s,
                        "source_split": split,
                        "dialect_name": _DIALECT_NAMES.get(dialect.lower()),
                        "timit_speaker_dir": speaker_dir.lower(),
                        "sentence_id": sentence_id.lower(),
                        "sentence_type": sentence_id[:2].lower(),
                    }
                )
                split_rows.append({"item_id": item_id, "split": split})
                phone_rows, skipped_phone_rows = _read_segmentation(
                    phn_path,
                    item_id=item_id,
                    interval_type=IntervalType.PHONE,
                    sample_rate=sample_rate,
                    duration_s=duration_s,
                )
                if skipped_phone_rows:
                    raise DatasetValidationError(
                        f"VTR phone segmentation unexpectedly contained zero-length intervals: {phn_path}"
                    )
                interval_rows.extend(phone_rows)

                word_rows, skipped_word_rows = _read_segmentation(
                    wrd_path,
                    item_id=item_id,
                    interval_type=IntervalType.WORD,
                    sample_rate=sample_rate,
                    duration_s=duration_s,
                    skip_zero_length=True,
                )
                skipped_zero_length_words += skipped_word_rows
                interval_rows.extend(word_rows)

        items = pd.DataFrame(item_rows)
        tracks = pd.concat(track_frames, ignore_index=True)
        intervals = pd.DataFrame(interval_rows)
        intervals = derive_vowel_intervals(intervals, parsed.vowel_labels)
        splits = pd.DataFrame(split_rows)

        manifest = DatasetManifest(
            name=parsed.name,
            source="vtr",
            adapter=self.name,
            adapter_version=self.version,
            annotation_type=AnnotationType.TRACK,
            available_formants=list(_VTR_AVAILABLE_FORMANTS),
            preparation_config={
                "vtr_root": str(vtr_root),
                "audio_root": str(audio_root),
                "language": parsed.language,
                "vowel_labels": parsed.vowel_labels,
                "frame_step_s": _VTR_FRAME_STEP_S,
                "source_htk_sample_periods": sorted(header_periods),
                "source_formants": [formant.value for formant in _VTR_AVAILABLE_FORMANTS],
                "source_f4_policy": "included_as_provided_evaluation_selection_is_user_controlled",
                "skipped_zero_length_word_intervals": skipped_zero_length_words,
            },
        )
        return PreparedDataset(
            manifest=manifest,
            items=items,
            tracks=tracks,
            intervals=intervals,
            splits=splits,
            static_measurements=None,
        )


def _resolve_split_roots(vtr_root: Path) -> dict[str, Path]:
    if not vtr_root.is_dir():
        raise DatasetValidationError(f"VTR vtr_root does not exist or is not a directory: {vtr_root}")
    children = {path.name.lower(): path for path in vtr_root.iterdir() if path.is_dir()}
    missing = [split for split in ("train", "test") if split not in children]
    if missing:
        raise DatasetValidationError(
            f"VTR root must contain Train and Test directories; missing: {', '.join(missing)}"
        )
    return {split: children[split] for split in ("train", "test")}


def _resolve_audio_path(
    audio_root: Path,
    *,
    split: str,
    dialect: str,
    speaker_dir: str,
    sentence_id: str,
) -> Path:
    candidates = (
        audio_root / split.upper() / dialect.upper() / speaker_dir.upper() / f"{sentence_id.upper()}.WAV",
        audio_root / split / dialect.lower() / speaker_dir.lower() / f"{sentence_id.lower()}.wav",
        audio_root / split / dialect / speaker_dir / f"{sentence_id}.wav",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise DatasetValidationError(
        "Missing TIMIT audio for VTR utterance. Expected a matching WAV such as: "
        f"{candidates[0]}"
    )


def _read_fb(path: Path, *, item_id: str) -> tuple[pd.DataFrame, int]:
    """Read one big-endian HTK VTR file and return canonical F1-F4 tracks in Hz."""
    payload = path.read_bytes()
    if len(payload) < _HTK_HEADER_SIZE:
        raise DatasetValidationError(f"VTR .fb file is too short to contain an HTK header: {path}")

    n_frames, sample_period, sample_size, _file_type = struct.unpack(">IIHH", payload[:_HTK_HEADER_SIZE])
    if sample_size % 4 != 0:
        raise DatasetValidationError(f"Invalid VTR HTK sample size in '{path}': {sample_size}")
    n_components = sample_size // 4
    if n_components < _VTR_COMPONENTS:
        raise DatasetValidationError(
            f"VTR .fb file '{path}' has {n_components} components; expected at least {_VTR_COMPONENTS}."
        )

    expected_values = n_frames * n_components
    values = np.frombuffer(payload, dtype=">f4", offset=_HTK_HEADER_SIZE)
    if len(values) != expected_values:
        raise DatasetValidationError(
            f"VTR .fb file '{path}' contains {len(values)} float values; expected {expected_values}."
        )
    data = values.reshape(n_frames, n_components)

    # Values are stored in kHz. Preserve all four formants supplied by VTR;
    # evaluation is responsible for selecting the formant subset to score.
    return (
        pd.DataFrame(
            {
                "item_id": item_id,
                "time_s": np.arange(n_frames, dtype=float) * _VTR_FRAME_STEP_S,
                "F1": data[:, 0].astype(float) * 1000.0,
                "F2": data[:, 1].astype(float) * 1000.0,
                "F3": data[:, 2].astype(float) * 1000.0,
                "F4": data[:, 3].astype(float) * 1000.0,
            }
        ),
        sample_period,
    )


def _read_segmentation(
    path: Path,
    *,
    item_id: str,
    interval_type: IntervalType,
    sample_rate: int,
    duration_s: float,
    skip_zero_length: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped_zero_length = 0
    for index, line in enumerate(path.read_text(encoding="ascii").splitlines()):
        if not line.strip():
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise DatasetValidationError(f"Malformed VTR segmentation line in '{path}': {line!r}")
        try:
            begin_sample, end_sample = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise DatasetValidationError(f"Non-integer VTR sample boundary in '{path}': {line!r}") from exc
        if begin_sample < 0 or end_sample < begin_sample:
            raise DatasetValidationError(f"Invalid VTR sample interval in '{path}': {line!r}")
        if end_sample == begin_sample:
            if skip_zero_length:
                skipped_zero_length += 1
                continue
            raise DatasetValidationError(f"Zero-length VTR sample interval in '{path}': {line!r}")

        start_s = begin_sample / sample_rate
        end_s = end_sample / sample_rate
        # TIMIT documentation notes that transcription endpoints can be slightly
        # shorter than the waveform. They must never extend beyond the waveform.
        if end_s > duration_s + 1e-6:
            raise DatasetValidationError(
                f"VTR segmentation '{path}' extends beyond its TIMIT waveform: {end_s:.6f}s > {duration_s:.6f}s."
            )
        rows.append(
            {
                "interval_id": f"{item_id}:{interval_type.value}:{index:04d}",
                "item_id": item_id,
                "interval_type": interval_type.value,
                "label": parts[2].strip(),
                "start_s": start_s,
                "end_s": end_s,
                "origin": IntervalOrigin.SOURCE.value,
            }
        )
    if not rows:
        raise DatasetValidationError(f"VTR segmentation file contains no usable intervals: {path}")
    return rows, skipped_zero_length


def _speaker_gender(speaker_dir: str) -> str:
    if not speaker_dir:
        raise DatasetValidationError("VTR speaker directory name is empty.")
    prefix = speaker_dir[0].lower()
    if prefix == "f":
        return "female"
    if prefix == "m":
        return "male"
    raise DatasetValidationError(f"VTR speaker directory must begin with f or m: {speaker_dir}")


def _read_audio_info(path: Path) -> tuple[int, int]:
    """Return ``(sample_rate, sample_count)`` for RIFF WAV or NIST SPHERE audio."""
    with path.open("rb") as file_obj:
        magic = file_obj.read(8)
    if magic.startswith(_SPHERE_MAGIC):
        return _read_sphere_audio_info(path)

    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_count = wav_file.getnframes()
    except (OSError, wave.Error) as exc:
        raise DatasetValidationError(f"Could not read TIMIT audio metadata from '{path}': {exc}") from exc
    if sample_rate <= 0 or sample_count <= 0:
        raise DatasetValidationError(f"Invalid WAV metadata in TIMIT audio: {path}")
    return sample_rate, sample_count


def _read_sphere_audio_info(path: Path) -> tuple[int, int]:
    try:
        with path.open("rb") as file_obj:
            first_line = file_obj.readline().decode("ascii").strip()
            header_size_line = file_obj.readline().decode("ascii").strip()
            header_size = int(header_size_line)
            file_obj.seek(0)
            header = file_obj.read(header_size).decode("ascii", errors="strict")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DatasetValidationError(f"Could not parse NIST SPHERE header from '{path}': {exc}") from exc

    if first_line != "NIST_1A" or header_size <= 0:
        raise DatasetValidationError(f"Invalid NIST SPHERE header in TIMIT audio: {path}")
    sample_rate = _sphere_integer(header, "sample_rate", path)
    sample_count = _sphere_integer(header, "sample_count", path)
    if sample_rate <= 0 or sample_count <= 0:
        raise DatasetValidationError(f"Invalid sample_rate/sample_count in TIMIT SPHERE audio: {path}")
    return sample_rate, sample_count


def _sphere_integer(header: str, field: str, path: Path) -> int:
    match = re.search(rf"(?m)^{re.escape(field)}\s+-i\s+(\d+)\s*$", header)
    if match is None:
        raise DatasetValidationError(f"NIST SPHERE header '{path}' is missing integer field '{field}'.")
    return int(match.group(1))
