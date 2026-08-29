"""Tests for VTR utterance-native preparation and TIMIT metadata handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from formant_benchmark.data.models import Formant
from formant_benchmark.datasets.vtr import VTRAdapter
from formant_benchmark.preparation.validation import validate_prepared_dataset
from tests.fixtures.vtr import create_vtr_layout


def _config(vtr_root: Path, audio_root: Path) -> dict[str, object]:
    return {
        "adapter": "vtr",
        "name": "vtr",
        "vtr_root": vtr_root,
        "audio_root": audio_root,
        "language": "english",
        "vowel_labels": ["iy"],
    }


def test_vtr_preserves_utterance_items_splits_and_intervals(tmp_path: Path) -> None:
    vtr_root, audio_root = create_vtr_layout(tmp_path)
    dataset = VTRAdapter().prepare(_config(vtr_root, audio_root))

    validate_prepared_dataset(dataset, require_audio=True)
    assert dataset.manifest.available_formants == [Formant.F1, Formant.F2, Formant.F3, Formant.F4]
    assert set(dataset.items["item_type"]) == {"utterance"}
    assert set(dataset.items["source_split"]) == {"train", "test"}
    assert set(dataset.splits["split"]) == {"train", "test"}
    assert dataset.items["speaker_id"].nunique() == 2
    assert set(dataset.intervals["interval_type"]) == {"phone", "word", "vowel"}

    vowel = dataset.intervals.loc[
        (dataset.intervals["interval_type"] == "vowel") & (dataset.intervals["label"] == "iy")
    ].iloc[0]
    assert vowel["origin"] == "derived"
    assert vowel["start_s"] == pytest.approx(0.01)
    assert vowel["end_s"] == pytest.approx(0.03)


def test_vtr_reads_htk_tracks_in_hz_and_preserves_f4(tmp_path: Path) -> None:
    vtr_root, audio_root = create_vtr_layout(tmp_path)
    dataset = VTRAdapter().prepare(_config(vtr_root, audio_root))
    first = dataset.tracks.loc[dataset.tracks["item_id"].str.contains("train")].reset_index(drop=True)

    assert first["time_s"].tolist() == pytest.approx([0.00, 0.01, 0.02, 0.03, 0.04])
    assert first["F1"].tolist() == pytest.approx([500, 510, 520, 530, 540])
    assert first["F2"].iloc[0] == pytest.approx(1500)
    assert first["F3"].iloc[0] == pytest.approx(2500)
    assert first["F4"].iloc[0] == pytest.approx(3500)
    assert dataset.manifest.preparation_config["source_formants"] == ["F1", "F2", "F3", "F4"]
    assert (
        dataset.manifest.preparation_config["source_f4_policy"]
        == "included_as_provided_evaluation_selection_is_user_controlled"
    )
    assert dataset.manifest.preparation_config["source_htk_sample_periods"] == [10000]


def test_vtr_parses_nist_sphere_audio_metadata(tmp_path: Path) -> None:
    vtr_root, audio_root = create_vtr_layout(tmp_path, sphere_audio=True)
    dataset = VTRAdapter().prepare(_config(vtr_root, audio_root))

    validate_prepared_dataset(dataset, require_audio=True)
    assert dataset.items["duration_s"].tolist() == pytest.approx([0.05, 0.05])


def test_vtr_default_inventory_does_not_make_syllabic_consonants_vowels(tmp_path: Path) -> None:
    vtr_root, audio_root = create_vtr_layout(tmp_path)
    phn = next((vtr_root / "Train").glob("dr*/*/*.phn"))
    phn.write_text("0 160 h#\n160 320 iy\n320 480 el\n480 800 t\n", encoding="ascii")

    dataset = VTRAdapter().prepare(
        {
            "adapter": "vtr",
            "name": "vtr",
            "vtr_root": vtr_root,
            "audio_root": audio_root,
        }
    )
    vowel_labels = set(dataset.intervals.loc[dataset.intervals["interval_type"] == "vowel", "label"])
    assert "iy" in vowel_labels
    assert "el" not in vowel_labels


def test_vtr_keeps_source_formants_without_acoustic_cleaning(tmp_path: Path) -> None:
    vtr_root, audio_root = create_vtr_layout(tmp_path)
    fb = next((vtr_root / "Train").glob("dr*/*/*.fb"))
    payload = bytearray(fb.read_bytes())
    # Replace the first F2 with 0.2 kHz while F1 remains 0.5 kHz. Structural
    # preparation should preserve this acoustically implausible source value.
    payload[16:20] = np.asarray([0.2], dtype=">f4").tobytes()
    fb.write_bytes(payload)

    dataset = VTRAdapter().prepare(_config(vtr_root, audio_root))
    train = dataset.tracks.loc[dataset.tracks["item_id"].str.contains("train")]
    assert train.iloc[0]["F2"] == pytest.approx(200.0)


def test_vtr_skips_zero_length_word_alignment_and_records_count(tmp_path: Path) -> None:
    vtr_root, audio_root = create_vtr_layout(tmp_path)
    wrd = next((vtr_root / "Train").glob("dr*/*/*.wrd"))
    wrd.write_text("160 480 eat\n480 480 it\n480 800 now\n", encoding="ascii")

    dataset = VTRAdapter().prepare(_config(vtr_root, audio_root))
    assert dataset.manifest.preparation_config["skipped_zero_length_word_intervals"] == 1
    train_words = dataset.intervals.loc[
        (dataset.intervals["item_id"].str.contains(":train:"))
        & (dataset.intervals["interval_type"] == "word")
    ]
    assert train_words["label"].tolist() == ["eat", "now"]


def test_vtr_vowel_inventory_order_is_canonicalized(tmp_path: Path) -> None:
    vtr_root, audio_root = create_vtr_layout(tmp_path)
    first = VTRAdapter().prepare(
        {
            "adapter": "vtr",
            "name": "vtr",
            "vtr_root": vtr_root,
            "audio_root": audio_root,
            "vowel_labels": ["iy", "ae"],
        }
    )
    second = VTRAdapter().prepare(
        {
            "adapter": "vtr",
            "name": "vtr",
            "vtr_root": vtr_root,
            "audio_root": audio_root,
            "vowel_labels": ["ae", "iy"],
        }
    )
    assert first.manifest.preparation_config["vowel_labels"] == ["ae", "iy"]
    assert second.manifest.preparation_config["vowel_labels"] == ["ae", "iy"]
