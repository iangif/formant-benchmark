"""Prepared-dataset selection, interval scoping, and WAV cropping."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pandas as pd

from formant_benchmark.data.models import (
    IntervalType,
    PreparedDataset,
    TrackingInputMode,
)
from formant_benchmark.exceptions import (
    ConfigurationError,
    UnsupportedTrackerConfigurationError,
)
from formant_benchmark.preparation.voicing import require_voiced_feature
from formant_benchmark.trackers.base import TrackerAdapter, TrackingInput


def build_tracking_inputs(
    dataset: PreparedDataset,
    tracker: TrackerAdapter,
    *,
    input_mode: TrackingInputMode,
    interval_type: str | None,
    split: str | None,
    temporary_directory: Path,
) -> list[TrackingInput]:
    """Select prepared items and construct deterministic tracker invocations."""
    capabilities = tracker.capabilities
    if input_mode not in capabilities.input_modes:
        raise UnsupportedTrackerConfigurationError(
            f"Tracker '{tracker.name}' does not support input mode '{input_mode.value}'."
        )
    if interval_type == IntervalType.VOICED.value:
        require_voiced_feature()
    if input_mode is TrackingInputMode.CROPPED_INTERVALS and not interval_type:
        raise ConfigurationError("cropped_intervals input requires --interval-type.")
    if interval_type and capabilities.interval_types and interval_type not in capabilities.interval_types:
        raise UnsupportedTrackerConfigurationError(
            f"Tracker '{tracker.name}' does not support interval type '{interval_type}'."
        )

    items = _select_items(dataset, split)
    needs_intervals = input_mode is not TrackingInputMode.FULL_ITEM
    intervals = dataset.intervals.iloc[0:0].copy()
    if needs_intervals:
        intervals = dataset.intervals
        if interval_type:
            intervals = intervals.loc[intervals["interval_type"].astype(str) == interval_type].copy()
        selected_ids = set(items["item_id"].astype(str))
        intervals = intervals.loc[intervals["item_id"].astype(str).isin(selected_ids)].copy()
    if input_mode is TrackingInputMode.CROPPED_INTERVALS and intervals.empty:
        raise ConfigurationError(f"No '{interval_type}' intervals are available for the selected items.")

    temporary_directory.mkdir(parents=True, exist_ok=True)
    result: list[TrackingInput] = []
    by_item = {str(key): group for key, group in intervals.groupby("item_id", sort=False)}
    empty_item_intervals = pd.DataFrame(columns=intervals.columns)
    if input_mode is TrackingInputMode.CROPPED_INTERVALS:
        _reject_overlapping_intervals(by_item)
    for item in items.sort_values("item_id", kind="stable").to_dict(orient="records"):
        item_id = str(item["item_id"])
        duration_s = float(item["duration_s"])
        audio_path = _resolve_audio_path(item.get("audio_path"), dataset.root)
        item_intervals = by_item.get(item_id, empty_item_intervals)
        if input_mode is TrackingInputMode.CROPPED_INTERVALS:
            for interval in item_intervals.sort_values(["start_s", "interval_id"], kind="stable").to_dict(orient="records"):
                start_s = float(interval["start_s"])
                end_s = float(interval["end_s"])
                input_unit_id = str(interval["interval_id"])
                cropped_path = temporary_directory / f"{_safe_name(input_unit_id)}.wav"
                _crop_wav(audio_path, cropped_path, start_s, end_s)
                result.append(
                    TrackingInput(
                        item_id=item_id,
                        input_unit_id=input_unit_id,
                        audio_path=cropped_path,
                        duration_s=end_s - start_s,
                        source_start_s=start_s,
                        source_end_s=end_s,
                        metadata=_metadata(item, interval),
                    )
                )
            continue

        if capabilities.requires_audio and not audio_path.is_file():
            raise ConfigurationError(f"Audio file does not exist for item '{item_id}': {audio_path}")
        serialized_intervals = tuple(_clean_record(record) for record in item_intervals.to_dict(orient="records"))
        result.append(
            TrackingInput(
                item_id=item_id,
                input_unit_id=item_id,
                audio_path=audio_path,
                duration_s=duration_s,
                source_start_s=0.0,
                source_end_s=duration_s,
                metadata=_metadata(item),
                intervals=serialized_intervals if input_mode is TrackingInputMode.FULL_ITEM_WITH_INTERVALS else (),
            )
        )
    return result


def _select_items(dataset: PreparedDataset, split: str | None) -> pd.DataFrame:
    if split is None:
        return dataset.items.copy()
    selected = dataset.splits.loc[dataset.splits["split"].astype(str) == split, "item_id"]
    if selected.empty:
        raise ConfigurationError(f"Prepared dataset contains no items in split '{split}'.")
    return dataset.items.loc[dataset.items["item_id"].astype(str).isin(set(selected.astype(str)))].copy()


def _resolve_audio_path(value: Any, root: Path | None) -> Path:
    if value is None or pd.isna(value):
        return Path("<missing-audio>")
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and root is not None:
        path = root / path
    return path


def _metadata(item: dict[str, Any], interval: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = _clean_record(item)
    if interval is not None:
        cleaned = _clean_record(interval)
        metadata.update(cleaned)
        if cleaned.get("interval_type") == IntervalType.VOWEL.value:
            metadata["vowel"] = cleaned.get("label")
    return metadata


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: (None if pd.isna(value) else value) for key, value in record.items()}


def _crop_wav(source: Path, destination: Path, start_s: float, end_s: float) -> None:
    if not source.is_file():
        raise ConfigurationError(f"Cannot crop missing audio file: {source}")
    try:
        with wave.open(str(source), "rb") as reader:
            params = reader.getparams()
            sample_rate = reader.getframerate()
            start_frame = max(0, round(start_s * sample_rate))
            end_frame = min(reader.getnframes(), round(end_s * sample_rate))
            if end_frame <= start_frame:
                raise ConfigurationError(f"Audio crop is empty for {source}: {start_s}-{end_s}s")
            reader.setpos(start_frame)
            frames = reader.readframes(end_frame - start_frame)
        with wave.open(str(destination), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(frames)
    except (wave.Error, OSError) as exc:
        raise ConfigurationError(f"Could not crop WAV audio '{source}': {exc}") from exc


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _reject_overlapping_intervals(by_item: dict[str, pd.DataFrame]) -> None:
    for item_id, frame in by_item.items():
        ordered = frame.sort_values(["start_s", "end_s"], kind="stable")
        previous_end: float | None = None
        for row in ordered.itertuples(index=False):
            start = float(row.start_s)
            end = float(row.end_s)
            if previous_end is not None and start < previous_end - 1e-9:
                raise ConfigurationError(
                    f"Selected cropped intervals overlap for item '{item_id}'; normalized predictions would be ambiguous."
                )
            previous_end = end
