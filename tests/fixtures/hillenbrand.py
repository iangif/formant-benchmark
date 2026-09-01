"""Tiny Hillenbrand-style fixture with real source naming and table structure."""

from __future__ import annotations

import wave
from pathlib import Path


def create_hillenbrand_layout(root: Path) -> Path:
    source = root / "Hillenbrand"
    for folder in ("men", "women", "kids"):
        (source / folder).mkdir(parents=True, exist_ok=True)

    tokens = {
        "m01ae": ("men", 0.65),
        "w01iy": ("women", 0.60),
        "b01ae": ("kids", 0.55),
        "g01uw": ("kids", 0.58),
    }
    for token, (folder, duration_s) in tokens.items():
        _write_wav(source / folder / f"{token}.wav", duration_s)

    header = """(c) 1995 James Hillenbrand

col1: filename
col2: duration in msec
...

"""
    rows = [
        _bigdata_row("m01ae", 323, 120, (650, 1700, 2500), zero_at=None),
        _bigdata_row("w01iy", 280, 220, (300, 2700, 3300), zero_at="p30_f3"),
        _bigdata_row("b01ae", 257, 238, (630, 2423, 3166), zero_at=None),
        _bigdata_row("g01uw", 260, 250, (400, 1100, 3000), zero_at="steady_f2"),
    ]
    (source / "bigdata.dat.txt").write_text(header + "\n".join(rows) + "\n", encoding="utf-8")

    timedata = """  Start = Start of vowel nucleus
    End = End of vowel nucleus
Center1 = Steady-state time from judge 1
Center2 = Steady-state time from judge 2

File   Start    End   Center1  Center2
m01ae  177.3   500.4   304.0    297.0
w01iy  110.0   390.0   205.0    214.0
b01ae  100.0   357.0   190.0    200.0
g01uw  120.0   380.0   220.0    230.0
"""
    (source / "timedata.dat.txt").write_text(timedata, encoding="utf-8")
    return source


def _bigdata_row(
    token: str,
    duration_ms: int,
    f0: int,
    steady: tuple[int, int, int],
    *,
    zero_at: str | None,
) -> str:
    values: list[int] = [duration_ms, f0, *steady]
    for position in range(10, 90, 10):
        f1 = steady[0] + position
        f2 = steady[1] - position
        f3 = steady[2] + position
        if zero_at == f"p{position}_f3":
            f3 = 0
        values.extend((f1, f2, f3))
    if zero_at == "steady_f2":
        values[3] = 0
    return token + " " + " ".join(str(value) for value in values)


def _write_wav(path: Path, duration_s: float, sample_rate: int = 16000) -> None:
    frame_count = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)
