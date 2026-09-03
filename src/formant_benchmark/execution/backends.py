"""Persistent local and container-compatible tracker execution backends."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from formant_benchmark.exceptions import ConfigurationError, TrackerExecutionError


@dataclass(frozen=True, slots=True)
class BackendResult:
    """One response from a persistent wrapper process."""

    returncode: int
    response: Mapping[str, Any] | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class ExecutionWorker(ABC):
    """One long-lived wrapper process serving sequential JSON requests."""

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """Whether this worker can accept another request."""

    @abstractmethod
    def request(self, payload: Mapping[str, Any], *, timeout_s: float | None) -> BackendResult:
        """Send one request and wait for its matching single-line response."""

    @abstractmethod
    def close(self) -> None:
        """Stop the process, gracefully when possible."""


_EOF = object()


class ProcessExecutionWorker(ExecutionWorker):
    """JSON-lines worker implemented over a subprocess's standard streams."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None,
        environment: Mapping[str, str],
    ) -> None:
        try:
            self._process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise TrackerExecutionError(f"Could not start tracker wrapper: {exc}") from exc
        if self._process.stdin is None or self._process.stdout is None or self._process.stderr is None:
            self._terminate()
            raise TrackerExecutionError("Tracker wrapper standard streams are unavailable.")
        self._stdin: TextIO = self._process.stdin
        self._responses: queue.Queue[str | object] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=200)
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._process.stdout,),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    @property
    def is_alive(self) -> bool:
        return self._process.poll() is None

    def request(self, payload: Mapping[str, Any], *, timeout_s: float | None) -> BackendResult:
        if not self.is_alive:
            return self._dead_result()
        try:
            self._stdin.write(json.dumps(dict(payload), separators=(",", ":"), default=_json_default) + "\n")
            self._stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return self._dead_result()
        try:
            line = self._responses.get(timeout=timeout_s)
        except queue.Empty:
            self._terminate()
            return BackendResult(-1, None, stderr=self._stderr_text(), timed_out=True)
        if line is _EOF:
            return self._dead_result()
        assert isinstance(line, str)
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            return BackendResult(self._returncode(default=0), None, stdout=line, stderr=self._stderr_text())
        if not isinstance(response, Mapping):
            return BackendResult(self._returncode(default=0), None, stdout=line, stderr=self._stderr_text())
        return BackendResult(0, response, stdout=line, stderr=self._stderr_text())

    def close(self) -> None:
        if self.is_alive:
            try:
                self._stdin.write('{"type":"shutdown"}\n')
                self._stdin.flush()
                self._stdin.close()
                self._process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired, ValueError):
                self._terminate()
        self._join_readers()

    def _read_stdout(self, stream: TextIO) -> None:
        try:
            for line in stream:
                if line.strip():
                    self._responses.put(line.rstrip("\r\n"))
        finally:
            self._responses.put(_EOF)

    def _read_stderr(self, stream: TextIO) -> None:
        for line in stream:
            self._stderr.append(line.rstrip("\r\n"))

    def _dead_result(self) -> BackendResult:
        try:
            self._process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
        return BackendResult(self._returncode(default=-1), None, stderr=self._stderr_text())

    def _returncode(self, *, default: int) -> int:
        value = self._process.poll()
        return default if value is None else value

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr)

    def _terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)

    def _join_readers(self) -> None:
        self._stdout_thread.join(timeout=1)
        self._stderr_thread.join(timeout=1)


class ExecutionBackend(ABC):
    """Runtime-neutral factory for persistent tracker workers."""

    name: str
    stages_inputs: bool = False

    def protocol_path(self, path: Path, work_root: Path) -> str:
        """Return an input path visible to the wrapper process."""
        return str(path)

    @abstractmethod
    def check(self, command: Sequence[str]) -> dict[str, Any]:
        """Verify configured prerequisites without installing anything."""

    @abstractmethod
    def start_worker(self, command: Sequence[str], work_root: Path) -> ExecutionWorker:
        """Start one wrapper that may serve the entire prediction run."""


class LocalExecutionBackend(ExecutionBackend):
    """Execute a user-managed local environment or executable."""

    name = "local"

    def __init__(
        self,
        *,
        working_directory: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.working_directory = Path(working_directory).expanduser() if working_directory else None
        self.environment = dict(environment or {})

    def check(self, command: Sequence[str]) -> dict[str, Any]:
        executable = _resolve_executable(command)
        problems: list[str] = []
        if executable is None:
            problems.append(f"Executable not found: {command[0] if command else '<empty command>'}")
        if self.working_directory is not None and not self.working_directory.is_dir():
            problems.append(f"Working directory does not exist: {self.working_directory}")
        return {
            "backend": self.name,
            "available": not problems,
            "executable": executable,
            "working_directory": str(self.working_directory) if self.working_directory else None,
            "problems": problems,
        }

    def start_worker(self, command: Sequence[str], work_root: Path) -> ExecutionWorker:
        check = self.check(command)
        if not check["available"]:
            raise TrackerExecutionError("; ".join(check["problems"]))
        environment = os.environ.copy()
        environment.update(self.environment)
        return ProcessExecutionWorker(
            [*command, "--stream"],
            cwd=self.working_directory,
            environment=environment,
        )


class ContainerExecutionBackend(ExecutionBackend):
    """Run one persistent wrapper through Docker or Apptainer."""

    name = "container"
    stages_inputs = True

    def __init__(self, *, runtime: str, image: str) -> None:
        if runtime not in {"docker", "apptainer"}:
            raise ConfigurationError("Container runtime must be 'docker' or 'apptainer'.")
        if not image:
            raise ConfigurationError("Container execution requires a non-empty image.")
        self.runtime = runtime
        self.image = image

    def check(self, command: Sequence[str]) -> dict[str, Any]:
        executable = shutil.which(self.runtime)
        problems = [] if executable else [f"Container runtime not found: {self.runtime}"]
        if not command:
            problems.append("Tracker wrapper command is empty.")
        return {
            "backend": self.name,
            "runtime": self.runtime,
            "image": self.image,
            "available": not problems,
            "executable": executable,
            "problems": problems,
        }

    def protocol_path(self, path: Path, work_root: Path) -> str:
        try:
            relative = path.resolve().relative_to(work_root.resolve())
        except ValueError as exc:
            raise TrackerExecutionError(f"Container input is outside its mounted work directory: {path}") from exc
        return "/work/" + relative.as_posix()

    def start_worker(self, command: Sequence[str], work_root: Path) -> ExecutionWorker:
        check = self.check(command)
        if not check["available"]:
            raise TrackerExecutionError("; ".join(check["problems"]))
        work = work_root.resolve()
        if self.runtime == "docker":
            argv = [
                "docker", "run", "--rm", "-i", "-v", f"{work}:/work", "-w", "/work",
                self.image, *command, "--stream",
            ]
        else:
            argv = [
                "apptainer", "exec", "--bind", f"{work}:/work", self.image,
                *command, "--stream",
            ]
        return ProcessExecutionWorker(argv, cwd=None, environment=os.environ.copy())


def backend_from_config(config: Mapping[str, Any]) -> ExecutionBackend:
    """Construct a backend from one tracker's execution mapping."""
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise ConfigurationError("Tracker 'execution' must be a mapping.")
    backend = execution.get("backend", "local")
    if backend == "local":
        environment = execution.get("environment", {})
        if not isinstance(environment, Mapping):
            raise ConfigurationError("execution.environment must be a mapping.")
        return LocalExecutionBackend(
            working_directory=execution.get("working_directory"),
            environment={str(key): str(value) for key, value in environment.items()},
        )
    if backend == "container":
        return ContainerExecutionBackend(
            runtime=str(execution.get("runtime", "docker")),
            image=str(execution.get("image", "")),
        )
    raise ConfigurationError(f"Unknown execution backend: '{backend}'.")


def _resolve_executable(command: Sequence[str]) -> str | None:
    if not command:
        return None
    candidate = Path(command[0]).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        return str(candidate) if candidate.is_file() else None
    return shutil.which(command[0])


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")
