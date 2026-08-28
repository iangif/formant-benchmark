"""Deterministic fingerprinting of normalized evaluation-relevant dataset content."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from formant_benchmark.data.models import PreparedDataset

_ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_MACHINE_SPECIFIC_ITEM_COLUMNS = frozenset({"audio_path"})
_LOCAL_PATH_KEYS = frozenset(
    {
        "audio_path",
        "audio_root",
        "data_root",
        "input_path",
        "output_path",
        "source_path",
        "source_root",
        "working_directory",
    }
)
_ORDER_INSENSITIVE_CONFIG_KEYS = frozenset({"available_formants", "batches", "vowels"})


def dataset_fingerprint(dataset: PreparedDataset) -> str:
    """Hash normalized content/config while ignoring machine-specific local paths.

    Dataframe row order is intentionally irrelevant. Common set-like preparation
    options (e.g. selected batches) are also canonicalized so equivalent prepared
    inputs do not acquire different identities from incidental configuration order.
    """
    manifest = dataset.manifest.model_dump(mode="json", exclude={"fingerprint"})
    manifest["preparation_config"] = _sanitize_config(manifest.get("preparation_config", {}))

    payload = {
        "manifest": manifest,
        "items": _canonical_rows(dataset.items, _MACHINE_SPECIFIC_ITEM_COLUMNS),
        "tracks": _canonical_rows(dataset.tracks),
        "intervals": _canonical_rows(dataset.intervals),
        "splits": _canonical_rows(dataset.splits),
        "static_measurements": _canonical_rows(dataset.static_measurements)
        if dataset.static_measurements is not None
        else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_rows(df: pd.DataFrame | None, excluded: frozenset[str] = frozenset()) -> list[str]:
    if df is None:
        return []
    columns = sorted(column for column in df.columns if column not in excluded)
    rows: list[str] = []
    for record in df.loc[:, columns].to_dict(orient="records"):
        normalized = {key: _canonical_scalar(value) for key, value in record.items()}
        rows.append(json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    rows.sort()
    return rows


def _canonical_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _sanitize_config(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _looks_like_local_path_key(key):
        return "<LOCAL_PATH>"
    if isinstance(value, Mapping):
        return {
            str(child_key): _sanitize_config(child_value, key=str(child_key))
            for child_key, child_value in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sanitized = [_sanitize_config(item) for item in value]
        if key in _ORDER_INSENSITIVE_CONFIG_KEYS and all(_is_scalar(item) for item in sanitized):
            return sorted(sanitized, key=lambda item: json.dumps(item, sort_keys=True))
        return sanitized
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and (value.startswith("/") or _ABSOLUTE_WINDOWS_PATH.match(value)):
        return "<ABSOLUTE_PATH>"
    return _canonical_scalar(value)


def _looks_like_local_path_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _LOCAL_PATH_KEYS or lowered.endswith(("_path", "_root", "_directory"))


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))
