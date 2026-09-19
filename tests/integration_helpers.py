"""Shared helpers for optional real-tracker integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from formant_benchmark.config.loading import load_yaml
from formant_benchmark.config.resolution import deep_merge
from formant_benchmark.execution.backends import ExecutionBackend, backend_from_config
from formant_benchmark.trackers.base import TrackerAdapter


def configured_real_tracker(
    tracker: TrackerAdapter,
) -> tuple[dict[str, Any], ExecutionBackend, tuple[str, ...]]:
    """Load, detect, and health-check one optional real tracker installation.

    A missing configured executable/working directory means the optional tracker is not
    installed on this machine, so the caller is skipped. Once those installation
    prerequisites are present, any tracker-specific health-check failure is a real test
    failure rather than a skip.
    """
    repository_root = Path(__file__).resolve().parents[1]
    config_path = repository_root / "configs" / "trackers" / f"{tracker.name}.yaml"
    config = load_yaml(config_path)
    effective = deep_merge(dict(tracker.default_configuration), config)

    backend = backend_from_config(effective)
    command = tuple(tracker.wrapper_command(effective))
    installation = backend.check(command)
    if not installation.get("available"):
        problems = installation.get("problems") or ["configured installation prerequisites are absent"]
        pytest.skip(
            f"Optional {tracker.name} installation not present according to {config_path.name}: "
            + "; ".join(str(problem) for problem in problems)
        )

    health = tracker.check_environment(effective)
    assert health.get("available") is True, (
        f"Configured {tracker.name} installation exists but failed its environment check:\n"
        + yaml.safe_dump(health, sort_keys=False)
    )
    return effective, backend, command
