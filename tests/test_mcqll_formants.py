"""Tests for the shared English/Japanese MCQLL Formants adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from formant_benchmark.data.models import Formant
from formant_benchmark.datasets.mcqll_formants import MCQLLFormantsAdapter
from formant_benchmark.preparation.validation import validate_prepared_dataset
from tests.fixtures.mcqll import create_source_layout


def _patch_parquet(monkeypatch: pytest.MonkeyPatch, frames: dict[Path, pd.DataFrame]) -> None:
    def fake_read_parquet(path: str | Path, *args, **kwargs) -> pd.DataFrame:
        return frames[Path(path)].copy(deep=True)

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)


@pytest.mark.parametrize(
    ("language", "corpus", "name"),
    [
        ("english", "ls_eng", "mcqll_english"),
        ("japanese", "gp_jpn", "mcqll_japanese"),
    ],
)
def test_shared_adapter_prepares_each_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    corpus: str,
    name: str,
) -> None:
    gold_root, audio_root, frames = create_source_layout(tmp_path, corpus=corpus, batches=("batch1",))
    _patch_parquet(monkeypatch, frames)

    dataset = MCQLLFormantsAdapter().prepare(
        {
            "adapter": "mcqll_formants",
            "name": name,
            "corpus": corpus,
            "language": language,
            "gold_root": gold_root,
            "audio_root": audio_root,
            "batches": "all",
        }
    )

    validate_prepared_dataset(dataset, require_audio=True)
    assert dataset.manifest.name == name
    assert dataset.manifest.adapter == "mcqll_formants"
    assert dataset.manifest.available_formants == [Formant.F1, Formant.F2, Formant.F3, Formant.F4]
    assert dataset.items.loc[0, "language"] == language
    assert dataset.items.loc[0, "item_type"] == "vowel"
    assert dataset.splits.empty
    assert set(dataset.intervals["interval_type"]) == {"phone", "vowel"}
    vowel = dataset.intervals.loc[dataset.intervals["interval_type"] == "vowel"].iloc[0]
    assert vowel["origin"] == "derived"
    assert vowel["start_s"] == pytest.approx(0.06)
    assert vowel["end_s"] == pytest.approx(0.14)


def test_batch_selection_preserves_batch_as_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gold_root, audio_root, frames = create_source_layout(
        tmp_path,
        corpus="ls_eng",
        batches=("batch1", "batch2"),
    )
    _patch_parquet(monkeypatch, frames)

    adapter = MCQLLFormantsAdapter()
    selected = adapter.prepare(
        {
            "adapter": "mcqll_formants",
            "name": "mcqll_english",
            "corpus": "ls_eng",
            "language": "english",
            "gold_root": gold_root,
            "audio_root": audio_root,
            "batches": ["batch2"],
        }
    )
    all_batches = adapter.prepare(
        {
            "adapter": "mcqll_formants",
            "name": "mcqll_english",
            "corpus": "ls_eng",
            "language": "english",
            "gold_root": gold_root,
            "audio_root": audio_root,
            "batches": "all",
        }
    )

    assert selected.items["batch"].tolist() == ["batch2"]
    assert selected.splits.empty
    assert set(all_batches.items["batch"]) == {"batch1", "batch2"}
    assert all_batches.manifest.preparation_config["batches"] == ["batch1", "batch2"]


def test_adapter_uses_raw_gold_and_excludes_nonexported_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gold_root, audio_root, frames = create_source_layout(tmp_path, corpus="ls_eng", batches=("batch1",))
    tracks_path = gold_root / "batch1" / "tracks.parquet"
    frames[tracks_path]["F1_s"] = [9999.0, 9999.0]
    _patch_parquet(monkeypatch, frames)

    dataset = MCQLLFormantsAdapter().prepare(
        {
            "adapter": "mcqll_formants",
            "name": "mcqll_english",
            "corpus": "ls_eng",
            "language": "english",
            "gold_root": gold_root,
            "audio_root": audio_root,
        }
    )

    assert len(dataset.items) == 1
    assert dataset.tracks["F1"].tolist() == [500.0, 510.0]
    assert not dataset.items["item_id"].str.contains("excluded").any()


def test_f4_is_inferred_as_unavailable_when_source_has_no_f4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gold_root, audio_root, frames = create_source_layout(
        tmp_path,
        corpus="gp_jpn",
        batches=("batch1",),
        include_f4=False,
    )
    _patch_parquet(monkeypatch, frames)
    dataset = MCQLLFormantsAdapter().prepare(
        {
            "adapter": "mcqll_formants",
            "name": "mcqll_japanese",
            "corpus": "gp_jpn",
            "language": "japanese",
            "gold_root": gold_root,
            "audio_root": audio_root,
        }
    )
    assert dataset.manifest.available_formants == [Formant.F1, Formant.F2, Formant.F3]


@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow is not installed")
def test_real_nested_parquet_source_round_trip(tmp_path: Path) -> None:
    """Exercise pandas/PyArrow nested-struct decoding when the engine is available."""
    gold_root, audio_root, frames = create_source_layout(tmp_path, corpus="ls_eng", batches=("batch1",))
    batch_root = gold_root / "batch1"
    frames[batch_root / "tokens.parquet"].to_parquet(batch_root / "tokens.parquet", index=False)
    frames[batch_root / "tracks.parquet"].to_parquet(batch_root / "tracks.parquet", index=False)

    dataset = MCQLLFormantsAdapter().prepare(
        {
            "adapter": "mcqll_formants",
            "name": "mcqll_english",
            "corpus": "ls_eng",
            "language": "english",
            "gold_root": gold_root,
            "audio_root": audio_root,
        }
    )
    validate_prepared_dataset(dataset, require_audio=True)
    assert len(dataset.items) == 1
