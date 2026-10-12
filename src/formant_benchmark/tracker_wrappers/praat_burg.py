"""Persistent Praat Burg wrapper using Parselmouth 0.4.7's bundled Praat."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
from collections.abc import Mapping
from typing import Any

from formant_benchmark.tracker_wrappers.streaming import serve_requests

EXPECTED_PARSELMOUTH_VERSION = "0.4.7"
_FORMANTS = ("F1", "F2", "F3", "F4")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stream", action="store_true", help="Serve JSON-lines requests on stdin/stdout.")
    mode.add_argument("--check", action="store_true", help="Check the installed Parselmouth version.")
    args = parser.parse_args()
    if args.check:
        return _check_environment()
    return serve_requests(_predict)


def _check_environment() -> int:
    try:
        version = importlib.metadata.version("praat-parselmouth")
        parselmouth = importlib.import_module("parselmouth")
    except Exception as exc:
        print(json.dumps({"parselmouth_version": None, "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "parselmouth_version": version,
                "praat_version": str(getattr(parselmouth, "PRAAT_VERSION", "unknown")),
            },
            sort_keys=True,
        )
    )
    return 0 if version == EXPECTED_PARSELMOUTH_VERSION else 2


def _predict(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    audio_path = request.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError("Praat Burg requires a non-empty audio_path.")
    parameters = request.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Praat Burg request parameters must be a mapping.")

    parselmouth = _load_parselmouth()
    sound = parselmouth.Sound(audio_path)
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()
    time_step = parameters.get("time_step")
    formant = sound.to_formant_burg(
        time_step=None if time_step is None else float(time_step),
        max_number_of_formants=float(parameters["max_number_of_formants"]),
        maximum_formant=float(parameters["maximum_formant"]),
        window_length=float(parameters["window_length"]),
        pre_emphasis_from=float(parameters["pre_emphasis_from"]),
    )

    # Querying exactly at each frame centre returns that frame's own values (no
    # interpolation from neighbours); formants Praat did not find come back as NaN.
    # F5 and above are dropped because the canonical vocabulary stops at F4.
    rows: list[dict[str, Any]] = []
    for time_s in formant.xs():
        numeric_time = float(time_s)
        if not math.isfinite(numeric_time):
            raise ValueError("Praat Burg returned a non-finite frame time.")
        row: dict[str, Any] = {"time_s": numeric_time}
        for number, name in enumerate(_FORMANTS, start=1):
            row[name] = _finite_or_none(formant.get_value_at_time(number, numeric_time))
        rows.append(row)
    return rows


def _load_parselmouth() -> Any:
    parselmouth = importlib.import_module("parselmouth")
    version = importlib.metadata.version("praat-parselmouth")
    if version != EXPECTED_PARSELMOUTH_VERSION:
        raise RuntimeError(
            f"Praat Burg wrapper requires praat-parselmouth {EXPECTED_PARSELMOUTH_VERSION}; found {version}."
        )
    return parselmouth


def _finite_or_none(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


if __name__ == "__main__":
    raise SystemExit(main())
