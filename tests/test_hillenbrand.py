"""Tests for Hillenbrand static-gold preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from formant_benchmark.data.models import AnnotationType, Formant
from formant_benchmark.datasets.hillenbrand import HillenbrandAdapter
from formant_benchmark.exceptions import DatasetValidationError
from formant_benchmark.preparation.validation import validate_prepared_dataset
from tests.fixtures.hillenbrand import create_hillenbrand_layout


def _config(root: Path) -> dict[str, object]:
    return {"adapter": "hillenbrand", "name": "hillenbrand", "root": root, "language": "english"}


def test_hillenbrand_prepares_static_items_intervals_and_measurements(tmp_path: Path) -> None:
    root = create_hillenbrand_layout(tmp_path)
    dataset = HillenbrandAdapter().prepare(_config(root))

    validate_prepared_dataset(dataset, require_audio=True)
    assert dataset.manifest.annotation_type is AnnotationType.STATIC
    assert dataset.manifest.available_formants == [Formant.F1, Formant.F2, Formant.F3]
    assert dataset.tracks.empty
    assert dataset.splits.empty
    assert len(dataset.items) == 4
    assert len(dataset.intervals) == 4
    assert len(dataset.static_measurements) == 4 * 9
    assert set(dataset.intervals["interval_type"]) == {"vowel"}
    assert set(dataset.intervals["origin"]) == {"source"}


def test_hillenbrand_preserves_all_source_positions(tmp_path: Path) -> None:
    root = create_hillenbrand_layout(tmp_path)
    dataset = HillenbrandAdapter().prepare(_config(root))
    static = dataset.static_measurements
    assert static is not None

    token = static.loc[static["item_id"] == "hillenbrand:m01ae"].sort_values("measurement_id")
    assert set(token["measurement_kind"]) == {"steady_state", "relative_position"}
    relative = sorted(token["relative_position"].dropna().tolist())
    assert relative == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    steady = token.loc[token["measurement_kind"] == "steady_state"].iloc[0]
    assert steady["time_s"] == pytest.approx(0.304)
    assert steady["steady_state_judge1_s"] == pytest.approx(0.304)
    assert steady["steady_state_judge2_s"] == pytest.approx(0.297)
    assert steady["steady_state_time_source"] == "center1"
    assert pd.isna(steady["relative_position"])


def test_hillenbrand_falls_back_to_center2_when_center1_is_outside_vowel(tmp_path: Path) -> None:
    root = create_hillenbrand_layout(tmp_path)
    timedata = root / "timedata.dat.txt"
    timedata.write_text(
        timedata.read_text().replace(
            "w01iy  110.0   390.0   205.0    214.0\n",
            "w01iy  110.0   390.0    90.0    214.0\n",
        )
    )

    dataset = HillenbrandAdapter().prepare(_config(root))
    static = dataset.static_measurements
    assert static is not None
    steady = static.loc[static["measurement_id"] == "hillenbrand:w01iy:steady_state"].iloc[0]

    assert steady["time_s"] == pytest.approx(0.214)
    assert steady["steady_state_time_source"] == "center2"
    assert dataset.manifest.preparation_config["steady_state_timing"] == {
        "center1_used": 3,
        "center2_fallback_used": 1,
        "steady_state_omitted": 0,
    }
    assert dataset.manifest.preparation_config["source_exceptions"] == []


def test_hillenbrand_omits_only_steady_state_when_both_centers_are_invalid(tmp_path: Path) -> None:
    root = create_hillenbrand_layout(tmp_path)
    timedata = root / "timedata.dat.txt"
    timedata.write_text(
        timedata.read_text().replace(
            "w01iy  110.0   390.0   205.0    214.0\n",
            "w01iy  110.0   390.0    90.0    410.0\n",
        )
    )

    dataset = HillenbrandAdapter().prepare(_config(root))
    static = dataset.static_measurements
    assert static is not None

    token_measurements = static.loc[static["item_id"] == "hillenbrand:w01iy"]
    assert len(token_measurements) == 8
    assert "hillenbrand:w01iy:steady_state" not in set(token_measurements["measurement_id"])
    assert sorted(token_measurements["relative_position"].dropna().tolist()) == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    assert "hillenbrand:w01iy" in set(dataset.items["item_id"])
    assert "hillenbrand:w01iy:vowel" in set(dataset.intervals["interval_id"])

    assert dataset.manifest.preparation_config["steady_state_timing"] == {
        "center1_used": 3,
        "center2_fallback_used": 0,
        "steady_state_omitted": 1,
    }
    assert dataset.manifest.preparation_config["source_exceptions"] == [
        {
            "source_id": "w01iy",
            "type": "no_valid_steady_state_time",
            "start_ms": 110.0,
            "end_ms": 390.0,
            "center1_ms": 90.0,
            "center2_ms": 410.0,
        }
    ]


def test_hillenbrand_maps_zero_formants_to_missing(tmp_path: Path) -> None:
    root = create_hillenbrand_layout(tmp_path)
    dataset = HillenbrandAdapter().prepare(_config(root))
    static = dataset.static_measurements
    assert static is not None

    missing_dynamic = static.loc[static["measurement_id"] == "hillenbrand:w01iy:p30"].iloc[0]
    assert pd.isna(missing_dynamic["F3"])
    missing_steady = static.loc[static["measurement_id"] == "hillenbrand:g01uw:steady_state"].iloc[0]
    assert pd.isna(missing_steady["F2"])
    assert static["F4"].isna().all()


def test_hillenbrand_decodes_filename_metadata(tmp_path: Path) -> None:
    root = create_hillenbrand_layout(tmp_path)
    dataset = HillenbrandAdapter().prepare(_config(root))
    items = dataset.items.set_index("item_id")

    assert items.loc["hillenbrand:m01ae", "speaker_id"] == "m01"
    assert items.loc["hillenbrand:m01ae", "speaker_group"] == "man"
    assert items.loc["hillenbrand:m01ae", "gender"] == "male"
    assert items.loc["hillenbrand:m01ae", "age_group"] == "adult"
    assert items.loc["hillenbrand:m01ae", "vowel"] == "ae"
    assert items.loc["hillenbrand:m01ae", "word"] == "had"
    assert items.loc["hillenbrand:g01uw", "speaker_group"] == "girl"
    assert items.loc["hillenbrand:g01uw", "age_group"] == "child"
    assert items.loc["hillenbrand:g01uw", "word"] == "who'd"


def test_hillenbrand_uses_wav_duration_not_vowel_duration_as_item_duration(tmp_path: Path) -> None:
    root = create_hillenbrand_layout(tmp_path)
    dataset = HillenbrandAdapter().prepare(_config(root))
    row = dataset.items.loc[dataset.items["item_id"] == "hillenbrand:m01ae"].iloc[0]

    assert row["duration_s"] == pytest.approx(0.65)
    assert row["source_vowel_duration_ms"] == pytest.approx(323)
    vowel = dataset.intervals.loc[dataset.intervals["item_id"] == "hillenbrand:m01ae"].iloc[0]
    assert vowel["start_s"] == pytest.approx(0.1773)
    assert vowel["end_s"] == pytest.approx(0.5004)


def test_hillenbrand_requires_matching_bigdata_and_timedata_tokens(tmp_path: Path) -> None:
    root = create_hillenbrand_layout(tmp_path)
    timedata = root / "timedata.dat.txt"
    timedata.write_text(timedata.read_text().replace("g01uw  120.0   380.0   220.0    230.0\n", ""))

    with pytest.raises(DatasetValidationError, match="token sets do not match"):
        HillenbrandAdapter().prepare(_config(root))


def test_hillenbrand_requires_audio_for_every_source_token(tmp_path: Path) -> None:
    root = create_hillenbrand_layout(tmp_path)
    (root / "men" / "m01ae.wav").unlink()

    with pytest.raises(DatasetValidationError, match="Missing Hillenbrand audio"):
        HillenbrandAdapter().prepare(_config(root))
