"""Dependency-free JSON-lines worker loop for tracker-side wrappers."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TextIO

PredictionHandler = Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]


def serve_requests(
    handler: PredictionHandler,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> int:
    """Serve requests until EOF/shutdown, isolating ordinary item failures."""
    source = input_stream or sys.stdin
    destination = output_stream or sys.stdout
    for line in source:
        if not line.strip():
            continue
        request: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("request must be a JSON object")
            if request.get("type") == "shutdown":
                return 0
            if request.get("type") != "request":
                raise ValueError("request type must be 'request'")
            if request.get("protocol_version") != "2":
                raise ValueError("unsupported wrapper protocol_version")
            input_unit_id = request.get("input_unit_id")
            if not isinstance(input_unit_id, str) or not input_unit_id:
                raise ValueError("request requires a non-empty input_unit_id")
            rows = [dict(row) for row in handler(request)]
            response = {
                "type": "result",
                "input_unit_id": input_unit_id,
                "rows": rows,
            }
        except Exception as exc:
            input_unit_id = request.get("input_unit_id") if isinstance(request, dict) else None
            response = {
                "type": "failure",
                "input_unit_id": input_unit_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        destination.write(json.dumps(response, separators=(",", ":"), default=_json_default) + "\n")
        destination.flush()
    return 0


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")
