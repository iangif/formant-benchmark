"""Prepare the Hillenbrand et al. (1995) American English vowel database."""

from __future__ import annotations

import re
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from formant_benchmark.data.models import AnnotationType, DatasetManifest, Formant, IntervalOrigin, IntervalType, PreparedDataset
from formant_benchmark.data.schemas import empty_splits, empty_tracks
from formant_benchmark.datasets.base import DatasetAdapter
from formant_benchmark.exceptions import ConfigurationError, DatasetValidationError

_FILENAME_RE = re.compile(r"^(?P<group>[mwbg])(?P<talker>\d{2})(?P<vowel>[a-z]{2})$")
_BIGDATA_ROW_RE = re.compile(r"^[mwbg]\d{2}[a-z]{2}\s")
_TIMEDATA_ROW_RE = re.compile(r"^[mwbg]\d{2}[a-z]{2}\s")
_PERCENT_POSITIONS = tuple(range(10, 90, 10))
_AVAILABLE_FORMANTS = (Formant.F1, Formant.F2, Formant.F3)

_GROUP_METADATA = {
    "m": {"speaker_group": "man", "gender": "male", "age_group": "adult", "audio_dir": "men"},
    "w": {"speaker_group": "woman", "gender": "female", "age_group": "adult", "audio_dir": "women"},
    "b": {"speaker_group": "boy", "gender": "male", "age_group": "child", "audio_dir": "kids"},
    "g": {"speaker_group": "girl", "gender": "female", "age_group": "child", "audio_dir": "kids"},
}

_VOWEL_WORDS = {
    "ae": "had",
    "ah": "hod",
    "aw": "hawed",
    "eh": "head",
    "er": "heard",
    "ey": "hayed",
    "ih": "hid",
    "iy": "heed",
    "oa": "hoed",
    "oo": "hood",
    "uh": "hud",
    "uw": "who'd",
}


class HillenbrandConfig(BaseModel):
    """Source-specific configuration for the Hillenbrand vowel database."""

    model_config = ConfigDict(extra="allow")

    adapter: str = "hillenbrand"
    name: str = Field(default="hillenbrand", min_length=1)
    root: Path
    bigdata_file: str = "bigdata.dat.txt"
    timedata_file: str = "timedata.dat.txt"
    language: str = Field(default="english", min_length=1)

    @field_validator("adapter")
    @classmethod
    def validate_adapter(cls, value: str) -> str:
        if value != "hillenbrand":
            raise ValueError("HillenbrandConfig requires adapter: hillenbrand")
        return value


class HillenbrandAdapter(DatasetAdapter):
    """Adapt Hillenbrand source-defined static formant observations.

    Each source WAV is one item. The manually identified vowel nucleus is a
    source interval. ``bigdata.dat`` contributes up to nine static observations
    per token: one steady-state point and eight points at 10%-80% of vowel
    duration. Source zeros in F1-F3 mean "not measurable" and are normalized to
    missing.
    """

    name = "hillenbrand"
    version = "1"

    def prepare(self, config: Mapping[str, Any]) -> PreparedDataset:
        try:
            parsed = HillenbrandConfig.model_validate(dict(config))
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid Hillenbrand dataset configuration: {exc}") from exc

        root = parsed.root.expanduser().resolve()
        if not root.is_dir():
            raise DatasetValidationError(f"Hillenbrand root does not exist or is not a directory: {root}")

        bigdata_path = root / parsed.bigdata_file
        timedata_path = root / parsed.timedata_file
        bigdata = _read_bigdata(bigdata_path)
        timedata = _read_timedata(timedata_path)
        _validate_source_identity(bigdata, timedata)

        item_rows: list[dict[str, Any]] = []
        interval_rows: list[dict[str, Any]] = []
        measurement_rows: list[dict[str, Any]] = []
        steady_state_counts = {
            "center1_used": 0,
            "center2_fallback_used": 0,
            "steady_state_omitted": 0,
        }
        source_exceptions: list[dict[str, Any]] = []

        for source_id in sorted(bigdata):
            measurement = bigdata[source_id]
            timing = timedata[source_id]
            parsed_id = _parse_source_id(source_id)
            group = _GROUP_METADATA[parsed_id["group"]]
            audio_path = _resolve_audio(root, group["audio_dir"], source_id)
            sample_rate, sample_count = _read_wav_info(audio_path)
            item_duration_s = sample_count / sample_rate

            start_s = timing["start_ms"] / 1000.0
            end_s = timing["end_ms"] / 1000.0
            center1_s = timing["center1_ms"] / 1000.0
            center2_s = timing["center2_ms"] / 1000.0
            _validate_vowel_timing(source_id, start_s, end_s, item_duration_s)
            steady_state_time_s, steady_state_time_source = _resolve_steady_state_time(
                start_s=start_s,
                end_s=end_s,
                center1_s=center1_s,
                center2_s=center2_s,
            )

            item_id = f"hillenbrand:{source_id}"
            interval_id = f"{item_id}:vowel"
            vowel_code = parsed_id["vowel"]
            item_rows.append(
                {
                    "item_id": item_id,
                    "source": "hillenbrand",
                    "item_type": "vowel_recording",
                    "speaker_id": source_id[:3],
                    "gender": group["gender"],
                    "age": None,
                    "language": parsed.language,
                    "dialect": None,
                    "audio_path": str(audio_path),
                    "duration_s": item_duration_s,
                    "speaker_group": group["speaker_group"],
                    "age_group": group["age_group"],
                    "source_file": source_id,
                    "vowel": vowel_code,
                    "word": _VOWEL_WORDS.get(vowel_code),
                    "source_vowel_duration_ms": measurement["duration_ms"],
                    "source_f0_steady_hz": _zero_to_missing(measurement["f0_steady"]),
                    "steady_state_judge1_s": center1_s,
                    "steady_state_judge2_s": center2_s,
                }
            )
            interval_rows.append(
                {
                    "interval_id": interval_id,
                    "item_id": item_id,
                    "interval_type": IntervalType.VOWEL.value,
                    "label": vowel_code,
                    "start_s": start_s,
                    "end_s": end_s,
                    "origin": IntervalOrigin.SOURCE.value,
                }
            )

            if steady_state_time_s is not None and steady_state_time_source is not None:
                steady_state_counts[
                    "center1_used" if steady_state_time_source == "center1" else "center2_fallback_used"
                ] += 1
                measurement_rows.append(
                    _static_row(
                        measurement_id=f"{item_id}:steady_state",
                        item_id=item_id,
                        interval_id=interval_id,
                        kind="steady_state",
                        relative_position=None,
                        time_s=steady_state_time_s,
                        values=measurement["steady"],
                        center1_s=center1_s,
                        center2_s=center2_s,
                        steady_state_time_source=steady_state_time_source,
                    )
                )
            else:
                steady_state_counts["steady_state_omitted"] += 1
                source_exceptions.append(
                    {
                        "source_id": source_id,
                        "type": "no_valid_steady_state_time",
                        "start_ms": timing["start_ms"],
                        "end_ms": timing["end_ms"],
                        "center1_ms": timing["center1_ms"],
                        "center2_ms": timing["center2_ms"],
                    }
                )

            for position in _PERCENT_POSITIONS:
                measurement_rows.append(
                    _static_row(
                        measurement_id=f"{item_id}:p{position:02d}",
                        item_id=item_id,
                        interval_id=interval_id,
                        kind="relative_position",
                        relative_position=position / 100.0,
                        time_s=None,
                        values=measurement[f"p{position}"],
                        center1_s=center1_s,
                        center2_s=center2_s,
                        steady_state_time_source=None,
                    )
                )

        items = pd.DataFrame(item_rows)
        intervals = pd.DataFrame(interval_rows)
        static = pd.DataFrame(measurement_rows)
        manifest = DatasetManifest(
            name=parsed.name,
            source="hillenbrand_1995",
            adapter=self.name,
            adapter_version=self.version,
            annotation_type=AnnotationType.STATIC,
            available_formants=list(_AVAILABLE_FORMANTS),
            preparation_config={
                "root": str(root),
                "bigdata_file": parsed.bigdata_file,
                "timedata_file": parsed.timedata_file,
                "language": parsed.language,
                "source_measurements": ["steady_state", *[f"{value}%" for value in _PERCENT_POSITIONS]],
                "zero_formant_policy": "source_zero_means_not_measurable_normalized_to_missing",
                "steady_state_time_policy": "center1_primary_center2_fallback_omit_if_neither_valid",
                "steady_state_timing": steady_state_counts,
                "source_exceptions": source_exceptions,
                "ignored_source_files": ["vowdata.dat.txt", "vowdata.ds.txt", "iddata.dat.txt", "misid.dat.txt"],
            },
        )
        return PreparedDataset(
            manifest=manifest,
            items=items,
            tracks=empty_tracks(),
            intervals=intervals,
            splits=empty_splits(),
            static_measurements=static,
        )


def _read_bigdata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise DatasetValidationError(f"Hillenbrand bigdata file does not exist: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip()
        if not _BIGDATA_ROW_RE.match(line):
            continue
        parts = line.split()
        if len(parts) != 30:
            raise DatasetValidationError(
                f"Hillenbrand bigdata row {line_number} must contain 30 fields; found {len(parts)}."
            )
        source_id = parts[0].lower()
        _parse_source_id(source_id)
        if source_id in rows:
            raise DatasetValidationError(f"Duplicate Hillenbrand bigdata token: {source_id}")
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise DatasetValidationError(f"Non-numeric Hillenbrand bigdata value on row {line_number}.") from exc
        row: dict[str, Any] = {
            "duration_ms": values[0],
            "f0_steady": values[1],
            "steady": tuple(values[2:5]),
        }
        offset = 5
        for position in _PERCENT_POSITIONS:
            row[f"p{position}"] = tuple(values[offset : offset + 3])
            offset += 3
        rows[source_id] = row
    if not rows:
        raise DatasetValidationError(f"Hillenbrand bigdata file contains no measurement rows: {path}")
    return rows


def _read_timedata(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise DatasetValidationError(f"Hillenbrand timedata file does not exist: {path}")
    rows: dict[str, dict[str, float]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip()
        if not _TIMEDATA_ROW_RE.match(line):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise DatasetValidationError(
                f"Hillenbrand timedata row {line_number} must contain 5 fields; found {len(parts)}."
            )
        source_id = parts[0].lower()
        _parse_source_id(source_id)
        if source_id in rows:
            raise DatasetValidationError(f"Duplicate Hillenbrand timedata token: {source_id}")
        try:
            start_ms, end_ms, center1_ms, center2_ms = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise DatasetValidationError(f"Non-numeric Hillenbrand timedata value on row {line_number}.") from exc
        rows[source_id] = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "center1_ms": center1_ms,
            "center2_ms": center2_ms,
        }
    if not rows:
        raise DatasetValidationError(f"Hillenbrand timedata file contains no timing rows: {path}")
    return rows


def _validate_source_identity(bigdata: Mapping[str, Any], timedata: Mapping[str, Any]) -> None:
    bigdata_ids = set(bigdata)
    timedata_ids = set(timedata)
    if bigdata_ids != timedata_ids:
        missing_timing = sorted(bigdata_ids - timedata_ids)
        missing_measurements = sorted(timedata_ids - bigdata_ids)
        raise DatasetValidationError(
            "Hillenbrand bigdata/timedata token sets do not match. "
            f"Missing timing: {missing_timing[:10]}; missing measurements: {missing_measurements[:10]}"
        )


def _parse_source_id(source_id: str) -> dict[str, str]:
    match = _FILENAME_RE.fullmatch(source_id.lower())
    if not match:
        raise DatasetValidationError(f"Invalid Hillenbrand filename/token identifier: {source_id}")
    return match.groupdict()


def _resolve_audio(root: Path, directory: str, source_id: str) -> Path:
    folder = root / directory
    candidates = (folder / f"{source_id}.wav", folder / f"{source_id}.WAV")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise DatasetValidationError(
        f"Missing Hillenbrand audio for '{source_id}'. Expected under '{folder}' with .wav/.WAV extension."
    )


def _read_wav_info(path: Path) -> tuple[int, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            sample_count = handle.getnframes()
    except (OSError, wave.Error) as exc:
        raise DatasetValidationError(f"Could not read Hillenbrand WAV '{path}': {exc}") from exc
    if sample_rate <= 0 or sample_count <= 0:
        raise DatasetValidationError(f"Hillenbrand WAV has invalid audio metadata: {path}")
    return sample_rate, sample_count


def _validate_vowel_timing(source_id: str, start_s: float, end_s: float, item_duration_s: float) -> None:
    """Validate the source vowel nucleus independently of steady-state judgments."""
    if start_s < 0 or end_s <= start_s or end_s > item_duration_s + 1e-6:
        raise DatasetValidationError(
            f"Invalid Hillenbrand vowel bounds for '{source_id}': start={start_s}, end={end_s}, "
            f"audio_duration={item_duration_s}."
        )


def _resolve_steady_state_time(
    *,
    start_s: float,
    end_s: float,
    center1_s: float,
    center2_s: float,
) -> tuple[float | None, str | None]:
    """Choose the first valid independent steady-state judgment.
    """
    if start_s <= center1_s <= end_s:
        return center1_s, "center1"
    if start_s <= center2_s <= end_s:
        return center2_s, "center2"
    return None, None


def _zero_to_missing(value: float) -> float:
    return np.nan if value == 0 else float(value)


def _static_row(
    *,
    measurement_id: str,
    item_id: str,
    interval_id: str,
    kind: str,
    relative_position: float | None,
    time_s: float | None,
    values: tuple[float, float, float],
    center1_s: float,
    center2_s: float,
    steady_state_time_source: str | None,
) -> dict[str, Any]:
    return {
        "measurement_id": measurement_id,
        "item_id": item_id,
        "interval_id": interval_id,
        "measurement_kind": kind,
        "relative_position": relative_position,
        "time_s": time_s,
        "window_start_s": None,
        "window_end_s": None,
        "F1": _zero_to_missing(values[0]),
        "F2": _zero_to_missing(values[1]),
        "F3": _zero_to_missing(values[2]),
        "F4": np.nan,
        "steady_state_judge1_s": center1_s,
        "steady_state_judge2_s": center2_s,
        "steady_state_time_source": steady_state_time_source,
    }
