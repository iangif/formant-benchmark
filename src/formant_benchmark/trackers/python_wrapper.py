"""Shared command and health-check plumbing for trackers run as isolated Python wrappers."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from formant_benchmark.exceptions import ConfigurationError
from formant_benchmark.execution.backends import LocalExecutionBackend, backend_from_config
from formant_benchmark.trackers.base import TrackerAdapter

CheckArguments = Callable[[Mapping[str, Any]], Sequence[str]]


def python_wrapper_command(
    config: Mapping[str, Any],
    *,
    module: str,
    label: str,
    environment: str,
) -> tuple[str, ...]:
    """Return ``execution.command`` followed by ``-m <module>``.

    *label* names the tracker in errors and *environment* describes the isolated
    environment whose Python ``execution.command`` must point to.
    """
    execution = config.get("execution", {})
    command = execution.get("command") if isinstance(execution, Mapping) else None
    if isinstance(command, str) and command.strip():
        prefix: tuple[str, ...] = (command,)
    elif isinstance(command, Sequence) and not isinstance(command, (str, bytes)) and command:
        prefix = tuple(str(part) for part in command)
    else:
        raise ConfigurationError(
            f"{label} requires execution.command pointing to the Python executable "
            f"inside its isolated {environment} environment."
        )
    return (*prefix, "-m", module)


def run_wrapper_check(
    tracker: TrackerAdapter,
    config: Mapping[str, Any],
    *,
    label: str,
    details: Mapping[str, Any] | None = None,
    check_arguments: CheckArguments | None = None,
    version_key: str | None = None,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Check execution prerequisites, then ask the wrapper's ``--check`` mode for package health.

    The wrapper must print one JSON object on stdout. When *version_key* is given, a
    reported version other than *expected_version* is the reported problem even if the
    wrapper also exited nonzero. *check_arguments* builds extra ``--check`` arguments
    from the configuration; a ``ConfigurationError`` it raises becomes a problem rather
    than an exception. Container backends skip the package check because the wrapper
    cannot be invoked outside a tracking run there.
    """
    backend = backend_from_config(config)
    command = tuple(tracker.wrapper_command(config))
    result = backend.check(command)
    result["tracker"] = tracker.name
    result.update(details or {})
    if not result.get("available"):
        return result

    if not isinstance(backend, LocalExecutionBackend):
        result["package_check"] = "not_run_for_container_backend"
        return result

    try:
        extra = tuple(check_arguments(config)) if check_arguments is not None else ()
    except ConfigurationError as exc:
        return _fail(result, str(exc))

    try:
        completed = subprocess.run(
            [*backend.resolve_command(command), "--check", *extra],
            cwd=backend.working_directory,
            env=backend.process_environment(),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _fail(result, f"{label} wrapper check failed: {exc}")

    try:
        reported = json.loads(completed.stdout.strip()) if completed.stdout.strip() else {}
    except json.JSONDecodeError:
        reported = {}
    if not isinstance(reported, dict):
        reported = {}
    result["package_check"] = reported

    observed_version = reported.get(version_key) if version_key else None
    if observed_version is not None and observed_version != expected_version:
        return _fail(
            result,
            f"{label} version mismatch: expected {expected_version}, found {observed_version}.",
        )
    if completed.returncode != 0:
        message = (
            reported.get("error")
            or completed.stderr.strip()
            or completed.stdout.strip()
            or "unknown wrapper check failure"
        )
        return _fail(result, f"{label} wrapper check failed: {message}")
    return result


def _fail(result: dict[str, Any], problem: str) -> dict[str, Any]:
    result["available"] = False
    result.setdefault("problems", []).append(problem)
    return result
