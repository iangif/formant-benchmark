"""Prepare final MCQLL gold-track exports into the benchmark dataset contract."""

from __future__ import annotations

import json
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from formant_benchmark.data.models import (
    AnnotationType,
    DatasetManifest,
    Formant,
    IntervalOrigin,
    IntervalType,
    PreparedDataset,
)
from formant_benchmark.data.schemas import FORMANT_COLUMNS, empty_splits
from formant_benchmark.datasets.base import DatasetAdapter
from formant_benchmark.exceptions import ConfigurationError, DatasetValidationError

_SOURCE_FORMANT_COLUMNS = {formant: formant for formant in FORMANT_COLUMNS}
_REQUIRED_EXPORT_FILES = ("tracks.parquet", "tokens.parquet", "export_manifest.json")
_TIME_TOLERANCE_S = 1e-3


class MCQLLFormantsConfig(BaseModel):
    """Source-specific configuration for a prepared MCQLL language dataset."""

    model_config = ConfigDict(extra="allow")

    adapter: str = "mcqll_formants"
    name: str = Field(min_length=1)
    corpus: str = Field(min_length=1)
    language: str = Field(min_length=1)
    gold_root: Path
    audio_root: Path
    batches: str | list[str] = "all"

    @field_validator("adapter")
    @classmethod
    def validate_adapter(cls, value: str) -> str:
        if value != "mcqll_formants":
            raise ValueError("MCQLLFormantsConfig requires adapter: mcqll_formants")
        return value

    @field_validator("batches")
    @classmethod
    def validate_batches(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            if value != "all":
                raise ValueError("batches must be 'all' or a non-empty list of batch names")
            return value
        cleaned = [str(batch).strip() for batch in value if str(batch).strip()]
        if not cleaned:
            raise ValueError("batches must be 'all' or a non-empty list of batch names")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("batches must not contain duplicates")
        return cleaned


class MCQLLFormantsAdapter(DatasetAdapter):
    """Adapt final formants-export snapshots for one MCQLL language.

    One exported vowel token becomes one source-native benchmark item. The adapter
    uses the exporter's raw ``F1``-``F4`` columns as canonical gold and retains
    annotation batches only as metadata, never as experimental splits.
    """

    name = "mcqll_formants"
    version = "1"

    def prepare(self, config: Mapping[str, Any]) -> PreparedDataset:
        try:
            parsed = MCQLLFormantsConfig.model_validate(dict(config))
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid MCQLL dataset configuration: {exc}") from exc

        gold_root = parsed.gold_root.expanduser().resolve()
        audio_root = parsed.audio_root.expanduser().resolve()
        batches = _resolve_batches(gold_root, parsed.batches)

        item_frames: list[pd.DataFrame] = []
        track_frames: list[pd.DataFrame] = []
        interval_frames: list[pd.DataFrame] = []
        source_schemas: dict[str, str] = {}

        for batch in batches:
            batch_root = gold_root / batch
            _require_export_files(batch_root)
            manifest = _load_export_manifest(batch_root / "export_manifest.json", parsed.corpus, batch)
            source_schemas[batch] = str(manifest.get("schema", "unknown"))

            tokens = pd.read_parquet(batch_root / "tokens.parquet")
            tracks = pd.read_parquet(batch_root / "tracks.parquet")
            items, normalized_tracks, intervals = _prepare_batch(
                tokens=tokens,
                tracks=tracks,
                corpus=parsed.corpus,
                language=parsed.language,
                batch=batch,
                audio_root=audio_root,
            )
            item_frames.append(items)
            track_frames.append(normalized_tracks)
            interval_frames.append(intervals)

        items = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
        tracks = pd.concat(track_frames, ignore_index=True) if track_frames else pd.DataFrame()
        intervals = pd.concat(interval_frames, ignore_index=True) if interval_frames else pd.DataFrame()

        if items["item_id"].duplicated().any():
            duplicates = sorted(items.loc[items["item_id"].duplicated(keep=False), "item_id"].astype(str).unique())
            raise DatasetValidationError(
                "MCQLL token_id values must be unique across selected batches. "
                f"Duplicate item_id values: {duplicates[:10]}"
            )

        available_formants = [
            Formant(formant)
            for formant in FORMANT_COLUMNS
            if formant in tracks.columns and tracks[formant].notna().any()
        ]
        if not available_formants:
            raise DatasetValidationError("Selected MCQLL batches contain no usable raw formant values.")

        preparation_config = {
            "corpus": parsed.corpus,
            "language": parsed.language,
            "batches": batches,
            "gold_root": str(gold_root),
            "audio_root": str(audio_root),
            "source_export_schemas": source_schemas,
            "gold_columns": dict(_SOURCE_FORMANT_COLUMNS),
        }
        manifest = DatasetManifest(
            name=parsed.name,
            source="mcqll",
            adapter=self.name,
            adapter_version=self.version,
            annotation_type=AnnotationType.TRACK,
            available_formants=available_formants,
            preparation_config=preparation_config,
        )
        return PreparedDataset(
            manifest=manifest,
            items=items,
            tracks=tracks,
            intervals=intervals,
            splits=empty_splits(),
            static_measurements=None,
        )


def _resolve_batches(gold_root: Path, configured: str | Sequence[str]) -> list[str]:
    if not gold_root.is_dir():
        raise DatasetValidationError(f"MCQLL gold_root does not exist or is not a directory: {gold_root}")
    if configured == "all":
        batches = sorted(
            path.name
            for path in gold_root.iterdir()
            if path.is_dir() and all((path / filename).is_file() for filename in _REQUIRED_EXPORT_FILES)
        )
        if not batches:
            raise DatasetValidationError(f"No complete MCQLL batch exports found under: {gold_root}")
        return batches

    batches = sorted(str(batch) for batch in configured)
    missing = [batch for batch in batches if not (gold_root / batch).is_dir()]
    if missing:
        raise DatasetValidationError(f"Requested MCQLL batch directory/directories do not exist: {missing}")
    return batches


def _require_export_files(batch_root: Path) -> None:
    missing = [name for name in _REQUIRED_EXPORT_FILES if not (batch_root / name).is_file()]
    if missing:
        raise DatasetValidationError(f"MCQLL batch '{batch_root.name}' is missing required export file(s): {missing}")


def _load_export_manifest(path: Path, corpus: str, batch: str) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Could not read MCQLL export manifest '{path}': {exc}") from exc
    if not isinstance(manifest, dict):
        raise DatasetValidationError(f"MCQLL export manifest must contain an object: {path}")

    manifest_corpus = manifest.get("corpus")
    manifest_batch = manifest.get("batch")
    if manifest_corpus not in (None, "ALL", corpus):
        raise DatasetValidationError(
            f"MCQLL export corpus mismatch for batch '{batch}': expected '{corpus}', found '{manifest_corpus}'."
        )
    if manifest_batch not in (None, "ALL", batch):
        raise DatasetValidationError(
            f"MCQLL export batch mismatch: directory '{batch}', manifest '{manifest_batch}'."
        )
    return manifest


def _prepare_batch(
    *,
    tokens: pd.DataFrame,
    tracks: pd.DataFrame,
    corpus: str,
    language: str,
    batch: str,
    audio_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(tokens, ("token_id", "corpus", "batch", "file_stem", "metadata", "export"), "tokens.parquet")
    _require_columns(tracks, ("token_id", "time", *_SOURCE_FORMANT_COLUMNS.values()), "tracks.parquet")

    statuses = tokens["export"].map(lambda value: _nested_get(value, "status"))
    exported = tokens.loc[statuses == "exported"].copy()
    if exported.empty:
        raise DatasetValidationError(f"MCQLL batch '{batch}' contains no exported gold tokens.")

    bad_corpus = exported["corpus"].astype(str) != corpus
    bad_batch = exported["batch"].astype(str) != batch
    if bad_corpus.any() or bad_batch.any():
        raise DatasetValidationError(
            f"MCQLL token metadata does not match configured corpus/batch for '{corpus}/{batch}'."
        )

    exported_ids = set(exported["token_id"].astype(str))
    source_track_ids = set(tracks["token_id"].astype(str))
    missing_tracks = sorted(exported_ids - source_track_ids)
    unexpected_tracks = sorted(source_track_ids - exported_ids)
    if missing_tracks:
        raise DatasetValidationError(
            f"MCQLL exported token(s) have no track rows in batch '{batch}': {missing_tracks[:10]}"
        )
    if unexpected_tracks:
        raise DatasetValidationError(
            f"MCQLL tracks contain token(s) not marked exported in batch '{batch}': {unexpected_tracks[:10]}"
        )

    item_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    timing: dict[str, tuple[float, float | None]] = {}

    for row in exported.to_dict(orient="records"):
        token_id = str(row["token_id"])
        file_stem = str(row["file_stem"])
        metadata = _as_mapping(row.get("metadata"), "tokens.parquet.metadata")
        audio_path = _audio_path(audio_root, batch, file_stem)
        duration_s = _wav_duration(audio_path)
        clip_begin = _optional_float(_nested_get(metadata, "intervals", "clip", "begin"))
        clip_end = _optional_float(_nested_get(metadata, "intervals", "clip", "end"))
        if clip_begin is not None and clip_end is not None and clip_end <= clip_begin:
            raise DatasetValidationError(f"Invalid clip bounds for MCQLL token '{token_id}'.")
        timing[token_id] = (duration_s, clip_begin)

        speaker = _optional_str(_nested_get(metadata, "speaker"))
        gender = _optional_str(_nested_get(metadata, "gender"))
        item_rows.append(
            {
                "item_id": token_id,
                "source": "mcqll",
                "item_type": "vowel",
                "speaker_id": speaker,
                "gender": gender,
                "age": None,
                "language": language,
                "dialect": None,
                "audio_path": str(audio_path),
                "duration_s": duration_s,
                "batch": batch,
                "corpus": corpus,
                "file_stem": file_stem,
                "discourse": _optional_str(_nested_get(metadata, "discourse")),
            }
        )

        phone_begin = _first_not_none(
            _optional_float(_nested_get(metadata, "intervals", "phone", "corrected_begin")),
            _optional_float(_nested_get(metadata, "intervals", "phone", "begin")),
        )
        phone_end = _first_not_none(
            _optional_float(_nested_get(metadata, "intervals", "phone", "corrected_end")),
            _optional_float(_nested_get(metadata, "intervals", "phone", "end")),
        )
        if phone_begin is None or phone_end is None:
            raise DatasetValidationError(f"MCQLL token '{token_id}' is missing phone boundaries.")
        start_s, end_s = _item_relative_bounds(
            phone_begin,
            phone_end,
            duration_s=duration_s,
            clip_begin=clip_begin,
            clip_end=clip_end,
            token_id=token_id,
        )

        linguistic = _as_mapping(metadata.get("linguistic"), "tokens.parquet.metadata.linguistic")
        phone = _optional_str(linguistic.get("phone"))
        ipa = _optional_str(linguistic.get("ipa"))
        common_interval_metadata = {
            "ipa": ipa,
            "word": _optional_str(linguistic.get("word")),
            "syllable": _optional_str(linguistic.get("syllable")),
            "transcription": _optional_str(linguistic.get("transcription")),
            "previous_phone": _optional_str(_nested_get(linguistic, "previous", "phone")),
            "previous_phone_ipa": _optional_str(_nested_get(linguistic, "previous", "ipa")),
            "following_phone": _optional_str(_nested_get(linguistic, "following", "phone")),
            "following_phone_ipa": _optional_str(_nested_get(linguistic, "following", "ipa")),
            "batch": batch,
        }
        label = phone or ipa or "vowel"
        interval_rows.append(
            {
                "interval_id": f"{token_id}:phone",
                "item_id": token_id,
                "interval_type": IntervalType.PHONE.value,
                "label": label,
                "start_s": start_s,
                "end_s": end_s,
                "origin": IntervalOrigin.SOURCE.value,
                **common_interval_metadata,
            }
        )
        interval_rows.append(
            {
                "interval_id": f"{token_id}:vowel",
                "item_id": token_id,
                "interval_type": IntervalType.VOWEL.value,
                "label": label,
                "start_s": start_s,
                "end_s": end_s,
                "origin": IntervalOrigin.DERIVED.value,
                **common_interval_metadata,
            }
        )

    normalized_track_frames: list[pd.DataFrame] = []
    for token_id, group in tracks.groupby("token_id", sort=False):
        item_id = str(token_id)
        duration_s, clip_begin = timing[item_id]
        source_times = pd.to_numeric(group["time"], errors="coerce")
        if source_times.isna().any():
            raise DatasetValidationError(f"MCQLL track time contains non-numeric values for token '{item_id}'.")
        time_s = _item_relative_track_times(source_times.astype(float), duration_s, clip_begin, item_id)
        normalized = pd.DataFrame({"item_id": item_id, "time_s": time_s})
        for formant, source_column in _SOURCE_FORMANT_COLUMNS.items():
            normalized[formant] = group[source_column].to_numpy(copy=True)
        normalized_track_frames.append(normalized)

    return (
        pd.DataFrame(item_rows),
        pd.concat(normalized_track_frames, ignore_index=True),
        pd.DataFrame(interval_rows),
    )


def _item_relative_track_times(
    times: pd.Series,
    duration_s: float,
    clip_begin: float | None,
    token_id: str,
) -> pd.Series:
    if ((times >= -_TIME_TOLERANCE_S) & (times <= duration_s + _TIME_TOLERANCE_S)).all():
        return times.clip(lower=0.0, upper=duration_s)
    if clip_begin is not None:
        shifted = times - clip_begin
        if ((shifted >= -_TIME_TOLERANCE_S) & (shifted <= duration_s + _TIME_TOLERANCE_S)).all():
            return shifted.clip(lower=0.0, upper=duration_s)
    raise DatasetValidationError(
        f"MCQLL track times for token '{token_id}' cannot be mapped into the local audio item duration."
    )


def _item_relative_bounds(
    begin: float,
    end: float,
    *,
    duration_s: float,
    clip_begin: float | None,
    clip_end: float | None,
    token_id: str,
) -> tuple[float, float]:
    if clip_begin is not None and clip_end is not None:
        if begin >= clip_begin - _TIME_TOLERANCE_S and end <= clip_end + _TIME_TOLERANCE_S:
            shifted_begin, shifted_end = begin - clip_begin, end - clip_begin
            if shifted_begin >= -_TIME_TOLERANCE_S and shifted_end <= duration_s + _TIME_TOLERANCE_S:
                return max(0.0, shifted_begin), min(duration_s, shifted_end)
    if begin >= -_TIME_TOLERANCE_S and end <= duration_s + _TIME_TOLERANCE_S:
        return max(0.0, begin), min(duration_s, end)
    raise DatasetValidationError(
        f"MCQLL phone bounds for token '{token_id}' cannot be mapped into the local audio item duration."
    )


def _audio_path(audio_root: Path, batch: str, file_stem: str) -> Path:
    candidates = (
        audio_root / batch / "audio" / f"{file_stem}.wav",
        audio_root / batch / f"{file_stem}.wav",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise DatasetValidationError(
        f"Missing MCQLL audio for '{batch}/{file_stem}'. Expected: {candidates[0]}"
    )


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                raise DatasetValidationError(f"Invalid WAV frame rate: {path}")
            return wav_file.getnframes() / frame_rate
    except (OSError, wave.Error) as exc:
        raise DatasetValidationError(f"Could not read MCQLL WAV file '{path}': {exc}") from exc


def _require_columns(df: pd.DataFrame, columns: Sequence[str], table: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise DatasetValidationError(f"MCQLL {table} is missing required column(s): {missing}")


def _nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _as_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"MCQLL {field} must contain structured metadata.")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _first_not_none(*values: float | None) -> float | None:
    return next((value for value in values if value is not None), None)
