"""Persistent deterministic wrapper implementing the streaming protocol."""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Mapping
from typing import Any

from formant_benchmark.tracker_wrappers.streaming import serve_requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="Serve JSON-lines requests on stdin/stdout.")
    args = parser.parse_args()
    if not args.stream:
        parser.error("the synthetic wrapper requires --stream")
    return serve_requests(_predict)


def _predict(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    parameters = request["parameters"]
    item_id = str(request["item_id"])
    if parameters.get("crash") or item_id in parameters.get("crash_item_ids", []):
        os._exit(3)
    hang_s = float(parameters.get("hang_s", 0.0))
    if hang_s > 0:
        time.sleep(hang_s)
    if parameters.get("fail") or item_id in parameters.get("fail_item_ids", []):
        raise RuntimeError("Intentional synthetic tracker failure.")
    if parameters.get("missing_output"):
        return []

    duration = float(request["duration_s"])
    step = float(parameters["frame_step_s"])
    count = max(1, math.floor(duration / step + 1e-9) + 1)
    times = [min(index * step, duration) for index in range(count)]
    if times[-1] < duration - 1e-9:
        times.append(duration)
    omitted = set(parameters.get("omit_formants", []))
    offset = float(parameters.get("offset_hz", 0.0))
    slope = float(parameters.get("slope_hz_per_s", 10.0))
    rows = []
    for index, time_s in enumerate(times):
        row: dict[str, Any] = {"item_id": item_id, "time_s": time_s}
        for formant in ("F1", "F2", "F3", "F4"):
            row[formant] = (
                None
                if formant in omitted
                else float(parameters[f"base_{formant.lower()}"]) + offset + slope * time_s
            )
        if parameters.get("missing_every") and index % int(parameters["missing_every"]) == 0:
            row["F1"] = None
        rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
