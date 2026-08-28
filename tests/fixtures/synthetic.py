"""Small deterministic prepared datasets covering trajectory and static gold."""

from __future__ import annotations

import pandas as pd

from formant_benchmark.data.models import AnnotationType, DatasetManifest, Formant, PreparedDataset
from formant_benchmark.data.schemas import empty_tracks


def trajectory_dataset(*, source_root: str = "/machine-a/synthetic") -> PreparedDataset:
    """Return two utterances with F1-F4 tracks, intervals, metadata, and splits."""
    items = pd.DataFrame(
        [
            {
                "item_id": "utt-1",
                "source": "synthetic",
                "item_type": "utterance",
                "speaker_id": "spk-a",
                "gender": "female",
                "age": 30,
                "language": "en",
                "dialect": "test-a",
                "audio_path": f"{source_root}/utt-1.wav",
                "duration_s": 0.4,
                "batch": "batch1",
            },
            {
                "item_id": "utt-2",
                "source": "synthetic",
                "item_type": "utterance",
                "speaker_id": "spk-b",
                "gender": "male",
                "age": 31,
                "language": "en",
                "dialect": "test-b",
                "audio_path": f"{source_root}/utt-2.wav",
                "duration_s": 0.4,
                "batch": "batch2",
            },
        ]
    )
    rows = []
    for item_index, item_id in enumerate(("utt-1", "utt-2")):
        for frame, time_s in enumerate((0.0, 0.1, 0.2, 0.3, 0.4)):
            rows.append(
                {
                    "item_id": item_id,
                    "time_s": time_s,
                    "F1": 500 + 10 * item_index + frame,
                    "F2": 1500 + 10 * item_index + frame,
                    "F3": 2500 + 10 * item_index + frame,
                    "F4": 3500 + 10 * item_index + frame,
                }
            )
    tracks = pd.DataFrame(rows)
    intervals = pd.DataFrame(
        [
            {
                "interval_id": "utt-1:p1",
                "item_id": "utt-1",
                "interval_type": "phone",
                "label": "iy",
                "start_s": 0.1,
                "end_s": 0.3,
                "origin": "source",
            },
            {
                "interval_id": "utt-1:v1",
                "item_id": "utt-1",
                "interval_type": "vowel",
                "label": "iy",
                "start_s": 0.1,
                "end_s": 0.3,
                "origin": "derived",
            },
            {
                "interval_id": "utt-2:p1",
                "item_id": "utt-2",
                "interval_type": "phone",
                "label": "aa",
                "start_s": 0.05,
                "end_s": 0.25,
                "origin": "source",
            },
        ]
    )
    splits = pd.DataFrame([{"item_id": "utt-1", "split": "train"}, {"item_id": "utt-2", "split": "test"}])
    manifest = DatasetManifest(
        name="synthetic_trajectory",
        source="synthetic",
        adapter="synthetic_fixture",
        annotation_type=AnnotationType.TRACK,
        available_formants=[Formant.F1, Formant.F2, Formant.F3, Formant.F4],
        preparation_config={"source_root": source_root, "vowels": ["iy", "aa"]},
    )
    return PreparedDataset(manifest=manifest, items=items, tracks=tracks, intervals=intervals, splits=splits)


def static_dataset(*, source_root: str = "/machine-a/static") -> PreparedDataset:
    """Return a static source with point, relative-position, and window observations."""
    items = pd.DataFrame(
        [
            {
                "item_id": "vowel-1",
                "source": "synthetic-static",
                "item_type": "vowel",
                "speaker_id": "spk-c",
                "gender": "female",
                "age": 22,
                "language": "en",
                "dialect": None,
                "audio_path": f"{source_root}/vowel-1.wav",
                "duration_s": 0.5,
            },
            {
                "item_id": "vowel-2",
                "source": "synthetic-static",
                "item_type": "vowel",
                "speaker_id": "spk-d",
                "gender": "male",
                "age": 24,
                "language": "en",
                "dialect": None,
                "audio_path": f"{source_root}/vowel-2.wav",
                "duration_s": 0.6,
            },
        ]
    )
    intervals = pd.DataFrame(
        [
            {
                "interval_id": "vowel-1:v1",
                "item_id": "vowel-1",
                "interval_type": "vowel",
                "label": "ae",
                "start_s": 0.0,
                "end_s": 0.5,
                "origin": "source",
            },
            {
                "interval_id": "vowel-2:v1",
                "item_id": "vowel-2",
                "interval_type": "vowel",
                "label": "iy",
                "start_s": 0.0,
                "end_s": 0.6,
                "origin": "source",
            },
        ]
    )
    static = pd.DataFrame(
        [
            {
                "measurement_id": "m-1",
                "item_id": "vowel-1",
                "interval_id": "vowel-1:v1",
                "measurement_kind": "point",
                "relative_position": 0.5,
                "time_s": None,
                "window_start_s": None,
                "window_end_s": None,
                "F1": 700.0,
                "F2": 1700.0,
                "F3": 2700.0,
                "F4": None,
            },
            {
                "measurement_id": "m-2",
                "item_id": "vowel-1",
                "interval_id": "vowel-1:v1",
                "measurement_kind": "window",
                "relative_position": None,
                "time_s": None,
                "window_start_s": 0.2,
                "window_end_s": 0.3,
                "F1": 710.0,
                "F2": 1710.0,
                "F3": 2710.0,
                "F4": None,
            },
            {
                "measurement_id": "m-3",
                "item_id": "vowel-2",
                "interval_id": "vowel-2:v1",
                "measurement_kind": "point",
                "relative_position": None,
                "time_s": 0.3,
                "window_start_s": None,
                "window_end_s": None,
                "F1": 300.0,
                "F2": 2300.0,
                "F3": 3100.0,
                "F4": None,
            },
        ]
    )
    splits = pd.DataFrame([{"item_id": "vowel-1", "split": "train"}, {"item_id": "vowel-2", "split": "test"}])
    manifest = DatasetManifest(
        name="synthetic_static",
        source="synthetic-static",
        adapter="synthetic_static_fixture",
        annotation_type=AnnotationType.STATIC,
        available_formants=[Formant.F1, Formant.F2, Formant.F3],
        preparation_config={"source_root": source_root},
    )
    return PreparedDataset(
        manifest=manifest,
        items=items,
        tracks=empty_tracks(),
        intervals=intervals,
        splits=splits,
        static_measurements=static,
    )
