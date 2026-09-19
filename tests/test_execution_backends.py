"""Unit tests for local checks and runtime-neutral container command construction."""

from __future__ import annotations

import os
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


def test_local_backend_resolves_repository_relative_execution_paths(
    tmp_path: Path, monkeypatch
) -> None:
    executable = tmp_path / ".venvs" / "tracker" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text("")
    working_directory = tmp_path / "external" / "Tracker"
    working_directory.mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "extra_python").mkdir()

    captured: dict[str, object] = {}

    def fake_worker(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        "formant_benchmark.execution.backends.ProcessExecutionWorker", fake_worker
    )
    backend = LocalExecutionBackend(
        working_directory="external/Tracker",
        environment={
            "PYTHONPATH": os.pathsep.join(["src", "extra_python"]),
            "UNCHANGED": "value",
        },
        repository_root=tmp_path,
    )

    check = backend.check([".venvs/tracker/bin/python", "-m", "wrapper"])
    assert check["available"] is True
    assert check["executable"] == str(executable.resolve())
    assert check["working_directory"] == str(working_directory.resolve())

    backend.start_worker([".venvs/tracker/bin/python", "-m", "wrapper"], tmp_path)
    assert captured["argv"] == [
        str(executable.resolve()),
        "-m",
        "wrapper",
        "--stream",
    ]
    kwargs = captured["kwargs"]
    assert kwargs["cwd"] == working_directory.resolve()
    environment = kwargs["environment"]
    assert environment["PYTHONPATH"] == os.pathsep.join(
        [str((tmp_path / "src").resolve()), str((tmp_path / "extra_python").resolve())]
    )
    assert environment["UNCHANGED"] == "value"


def test_local_backend_leaves_bare_commands_for_system_path(tmp_path: Path) -> None:
    backend = LocalExecutionBackend(repository_root=tmp_path)
    assert backend.resolve_command(["python", "-m", "wrapper"]) == (
        "python",
        "-m",
        "wrapper",
    )


def test_local_backend_preserves_absolute_python_executable(tmp_path: Path) -> None:
    backend = LocalExecutionBackend(repository_root=tmp_path)
    assert backend.resolve_command([sys.executable, "-m", "wrapper"]) == (
        sys.executable,
        "-m",
        "wrapper",
    )


def test_docker_backend_mounts_protocol_work_directory(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_worker(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        "formant_benchmark.execution.backends.shutil.which", lambda _name: "/usr/bin/docker"
    )
    monkeypatch.setattr(
        "formant_benchmark.execution.backends.ProcessExecutionWorker", fake_worker
    )
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
