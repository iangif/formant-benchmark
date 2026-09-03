"""Unit tests for local checks and runtime-neutral container command construction."""

from __future__ import annotations

import sys
from pathlib import Path

from formant_benchmark.execution.backends import (
    ContainerExecutionBackend,
    LocalExecutionBackend,
)


def test_local_backend_check_finds_explicit_python() -> None:
    result = LocalExecutionBackend().check([sys.executable, "-m", "example"])
    assert result["available"] is True
    assert result["executable"] == sys.executable


def test_docker_backend_mounts_protocol_work_directory(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_worker(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("formant_benchmark.execution.backends.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("formant_benchmark.execution.backends.ProcessExecutionWorker", fake_worker)
    backend = ContainerExecutionBackend(runtime="docker", image="tracker:test")
    backend.start_worker(["python", "wrapper.py"], tmp_path)

    assert captured["argv"] == [
        "docker",
        "run",
        "--rm",
        "-i",
        "-v",
        f"{tmp_path.resolve()}:/work",
        "-w",
        "/work",
        "tracker:test",
        "python",
        "wrapper.py",
        "--stream",
    ]
