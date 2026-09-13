"""Persistent FastTrackPy 0.6.1 wrapper emitting smoothed winner trajectories."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from formant_benchmark.tracker_wrappers.streaming import serve_requests

EXPECTED_FASTTRACKPY_VERSION = "0.6.1"

_HEURISTIC_ATTRIBUTES = {
    "f1_max": "F1_Max",
    "f4_min": "F4_Min",
    "b2_max": "B2_Max",
    "b3_max": "B3_Max",
    "rhotic": "Rhotic",
    "f3_f4_sep": "F3_F4_Sep",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stream", action="store_true", help="Serve JSON-lines requests on stdin/stdout.")
    mode.add_argument("--check", action="store_true", help="Check the installed FastTrackPy version.")
    args = parser.parse_args()
    if args.check:
        return _check_environment()
    return serve_requests(_predict)


def _check_environment() -> int:
    try:
        version = importlib.metadata.version("fasttrackpy")
        importlib.import_module("fasttrackpy")
    except Exception as exc:
        print(json.dumps({"fasttrackpy_version": None, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"fasttrackpy_version": version}, sort_keys=True))
    return 0 if version == EXPECTED_FASTTRACKPY_VERSION else 2


def _predict(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    audio_path = request.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError("FastTrackPy requires a non-empty audio_path.")
    parameters = request.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("FastTrackPy request parameters must be a mapping.")

    api = _load_api()
    candidates = api["process_audio_file"](
        audio_path,
        min_max_formant=float(parameters["min_max_formant"]),
        max_max_formant=float(parameters["max_max_formant"]),
        nstep=int(parameters["nstep"]),
        n_formants=int(parameters["n_formants"]),
        window_length=float(parameters["window_length"]),
        time_step=float(parameters["time_step"]),
        pre_emphasis_from=float(parameters["pre_emphasis_from"]),
        pitch_floor=float(parameters["pitch_floor"]),
        smoother=api["Smoother"](
            method=str(parameters["smoother_method"]),
            order=int(parameters["smoother_order"]),
        ),
        loss_fun=api["Loss"](method=str(parameters["loss_method"])),
        agg_fun=api["Agg"](method=str(parameters["agg_method"])),
        heuristics=_resolve_heuristics(parameters.get("heuristics", [])),
    )
    winner = candidates.winner
    times = _to_vector(winner.time_domain, "winner.time_domain")
    smooth = _to_matrix(winner.smoothed_formants, "winner.smoothed_formants")
    if len(smooth) < 1 or len(smooth) > 4:
        raise ValueError(f"FastTrackPy winner returned {len(smooth)} formants; expected 1-4.")
    if any(len(row) != len(times) for row in smooth):
        raise ValueError("FastTrackPy winner time and smoothed-formant lengths do not match.")

    rows: list[dict[str, Any]] = []
    for index, time_s in enumerate(times):
        numeric_time = float(time_s)
        if not math.isfinite(numeric_time):
            raise ValueError("FastTrackPy winner returned a non-finite timestamp.")
        row: dict[str, Any] = {"time_s": numeric_time}
        for formant_index, formant in enumerate(("F1", "F2", "F3", "F4")):
            row[formant] = (
                _finite_or_none(smooth[formant_index][index])
                if formant_index < len(smooth)
                else None
            )
        rows.append(row)
    return rows


def _load_api() -> dict[str, Any]:
    fasttrackpy = importlib.import_module("fasttrackpy")
    version = importlib.metadata.version("fasttrackpy")
    if version != EXPECTED_FASTTRACKPY_VERSION:
        raise RuntimeError(
            f"FastTrackPy wrapper requires version {EXPECTED_FASTTRACKPY_VERSION}; found {version}."
        )
    return {
        "process_audio_file": getattr(fasttrackpy, "process_audio_file"),
        "Smoother": _resolve_symbol("Smoother", "fasttrackpy.processors.smoothers"),
        "Loss": _resolve_symbol("Loss", "fasttrackpy.processors.losses"),
        "Agg": _resolve_symbol("Agg", "fasttrackpy.processors.aggs"),
    }


def _resolve_symbol(name: str, module_name: str) -> Any:
    fasttrackpy = importlib.import_module("fasttrackpy")
    if hasattr(fasttrackpy, name):
        return getattr(fasttrackpy, name)
    module = importlib.import_module(module_name)
    return getattr(module, name)


def _resolve_heuristics(names: Any) -> list[Any]:
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise ValueError("FastTrackPy heuristics must be a list of names.")
    if not names:
        return []
    namespace = _heuristic_namespace()
    result = []
    for name in names:
        attribute = _HEURISTIC_ATTRIBUTES.get(str(name))
        if attribute is None:
            raise ValueError(f"Unknown FastTrackPy heuristic: {name}")
        result.append(getattr(namespace, attribute))
    return result


def _heuristic_namespace() -> Any:
    processors = importlib.import_module("fasttrackpy.processors")
    if hasattr(processors, "heuristic"):
        return getattr(processors, "heuristic")
    for module_name in ("fasttrackpy.processors.heuristic", "fasttrackpy.processors.heuristics"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ImportError("Could not import FastTrackPy's pre-specified heuristic namespace.")



def _finite_or_none(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None

def _to_vector(value: Any, name: str) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"FastTrackPy {name} is not one-dimensional sequence data.")
    return [float(item) for item in value]


def _to_matrix(value: Any, name: str) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"FastTrackPy {name} is not matrix data.")
    result: list[list[float]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"FastTrackPy {name} is not matrix data.")
        result.append([float(item) for item in row])
    return result


if __name__ == "__main__":
    raise SystemExit(main())
