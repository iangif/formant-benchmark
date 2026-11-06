"""Pages of example vowels with gold and predicted tracks, for checking results by eye."""

from __future__ import annotations

import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from formant_benchmark.analysis.load import load_unit_metrics
from formant_benchmark.analysis.plots import (
    _LINESTYLES,
    _STYLE,
    _SURFACE,
    _TEXT,
    _TEXT_SECONDARY,
    TRACKER_LABELS,
    _dataset_label,
    _datasets,
    _series,
    _slot,
    _trackers,
    _tracker_handles,
)
from formant_benchmark.analysis.positions import _run_root
from formant_benchmark.config.discovery import EVALUATIONS_ROOT, PREPARED_ROOT, RUNS_ROOT
from formant_benchmark.data.io import load_prepared_dataset
from formant_benchmark.data.models import AnnotationType
from formant_benchmark.runs.io import load_prediction_run

EXAMPLES_ROOT = Path("reports/examples")
SPECTROGRAM_CHOICES = ("worst", "all", "none")

_ROWS, _COLUMNS = 4, 5
_CONTEXT_S = 0.03
_FORMANTS = ("F1", "F2", "F3")
_SHORT_LABELS = {"praat_burg": "Praat", "fasttrackpy": "FTP", "formants_tracker": "FT"}
_SPECTROGRAM = LinearSegmentedColormap.from_list("spectrogram", [_SURFACE, "#6f6e69"])


@dataclass(frozen=True)
class _Section:
    title: str
    description: str
    intervals: list[str]
    spectrogram: bool


def write_examples(
    *,
    evaluations_root: str | Path = EVALUATIONS_ROOT,
    prepared_root: str | Path = PREPARED_ROOT,
    runs_root: str | Path = RUNS_ROOT,
    output: str | Path = EXAMPLES_ROOT,
    count: int = 20,
    seed: int = 0,
    spectrograms: str = "all",
    datasets: Sequence[str] | None = None,
) -> list[Path]:
    """Write ``examples_<dataset>.pdf`` and ``.csv`` per dataset and return the written paths.

    Each PDF has sections of *count* vowels: a seeded random sample, the vowels with
    the highest MAE for each tracker, and the vowels where the trackers' mean
    predictions disagree most. A vowel's score is its mean MAE over F1-F3 (and over
    its measurement points for static gold). Every panel shows all trackers.
    *spectrograms* puts the audio's spectrogram behind the tracks on every section
    ("all"), on the worst-vowel sections only ("worst"), or nowhere ("none"). Predictions
    outside the vowel are drawn faded: they are not evaluated, and in datasets with
    several vowels per item they can come from a neighbouring vowel's crop. The CSV lists
    each panel's section, ids, and per-tracker MAE in page order.
    """
    if spectrograms not in SPECTROGRAM_CHOICES:
        raise ValueError(f"spectrograms must be one of: {', '.join(SPECTROGRAM_CHOICES)}")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    units = load_unit_metrics(evaluations_root, formants=_FORMANTS)
    if datasets:
        units = units.loc[units["dataset"].isin(datasets)]
    written: list[Path] = []
    for dataset_name in _datasets(units):
        dataset_units = units.loc[units["dataset"] == dataset_name]
        dataset = load_prepared_dataset(Path(prepared_root) / dataset_name)
        predictions = {
            str(tracker): load_prediction_run(_run_root(Path(runs_root), str(group["prediction_run_id"].iloc[0]))).predictions
            for tracker, group in dataset_units.groupby("tracker", sort=False)
        }
        sections = _sections(dataset_units, count=count, seed=seed, spectrograms=spectrograms)
        path = destination / f"examples_{dataset_name}.pdf"
        with PdfPages(path) as pdf:
            for section in sections:
                for page, start in enumerate(range(0, len(section.intervals), _ROWS * _COLUMNS)):
                    figure = _page(
                        dataset, dataset_units, predictions, section,
                        section.intervals[start:start + _ROWS * _COLUMNS], page,
                    )
                    pdf.savefig(figure)
                    plt.close(figure)
        written.append(path)
        listing = destination / f"examples_{dataset_name}.csv"
        _listing(dataset_units, sections).to_csv(listing, index=False)
        written.append(listing)
    return written


def _listing(units: pd.DataFrame, sections: list[_Section]) -> pd.DataFrame:
    scores = units.groupby(["interval_id", "tracker"])["mae"].mean().unstack("tracker").add_prefix("mae_")
    first = units.drop_duplicates("interval_id").set_index("interval_id")[["item_id", "speaker_id", "vowel"]]
    rows = [
        {"section": section.title, "panel": position + 1, "interval_id": interval_id}
        for section in sections
        for position, interval_id in enumerate(section.intervals)
    ]
    frame = pd.DataFrame(rows, columns=["section", "panel", "interval_id"])
    return frame.join(first, on="interval_id").join(scores, on="interval_id")


def _sections(units: pd.DataFrame, *, count: int, seed: int, spectrograms: str) -> list[_Section]:
    scores = units.groupby(["interval_id", "tracker"])["mae"].mean().unstack("tracker")
    intervals = scores.index.to_series()
    sections = [
        _Section(
            "Random",
            f"{min(count, len(intervals))} vowels drawn at random (seed {seed})",
            intervals.sample(n=min(count, len(intervals)), random_state=seed).tolist(),
            spectrograms == "all",
        )
    ]
    for tracker in _trackers(units):
        worst = scores[tracker].dropna().sort_values(ascending=False).head(count)
        sections.append(
            _Section(
                f"Worst {TRACKER_LABELS.get(tracker, tracker)}",
                f"The {len(worst)} vowels with the highest {TRACKER_LABELS.get(tracker, tracker)} MAE (mean over F1-F3)",
                worst.index.tolist(),
                spectrograms in ("worst", "all"),
            )
        )
    if scores.shape[1] > 1:
        means = units.groupby(["interval_id", "formant", "tracker"])["pred_mean"].mean().unstack("tracker")
        spread = (means.max(axis=1) - means.min(axis=1)).groupby(level="interval_id").mean()
        spread = spread.dropna().sort_values(ascending=False).head(count)
        sections.append(
            _Section(
                "Disagreement",
                f"The {len(spread)} vowels where the trackers' mean predictions differ most (max - min, mean over F1-F3)",
                spread.index.tolist(),
                spectrograms == "all",
            )
        )
    return sections


def _page(dataset, units: pd.DataFrame, predictions: dict[str, pd.DataFrame], section: _Section,
          interval_ids: list[str], page: int) -> Figure:
    trackers = _trackers(units)
    scores = units.groupby(["interval_id", "tracker"])["mae"].mean()
    first = units.drop_duplicates("interval_id").set_index("interval_id")
    intervals = dataset.intervals.set_index("interval_id")
    items = dataset.items.set_index("item_id")
    static = dataset.manifest.annotation_type is AnnotationType.STATIC
    with plt.rc_context(_STYLE | {"font.size": 6.5, "axes.titlesize": 6.5}):
        fig, axes = plt.subplots(_ROWS, _COLUMNS, figsize=(11, 8.5), layout="constrained", squeeze=False)
        for ax, interval_id in zip(axes.flat, interval_ids, strict=False):
            interval = intervals.loc[interval_id]
            item_id = str(interval["item_id"])
            start_s, end_s = float(interval["start_s"]), float(interval["end_s"])
            window = (max(0.0, start_s - _CONTEXT_S), end_s + _CONTEXT_S)
            top = _top_frequency(dataset, interval_id, item_id, start_s, end_s, static, predictions, window)
            if section.spectrogram:
                _draw_spectrogram(ax, _audio_path(items.loc[item_id, "audio_path"], dataset.root), window, start_s, top)
            for tracker in trackers:
                _draw_prediction(ax, predictions[tracker], item_id, window, start_s, end_s, tracker)
            _draw_gold(ax, dataset, interval_id, item_id, start_s, end_s, static)
            for boundary in (0.0, (end_s - start_s) * 1000):
                ax.axvline(boundary, color=_TEXT_SECONDARY, linewidth=0.6, linestyle=(0, (2, 2)))
            ax.set_xlim((window[0] - start_s) * 1000, (window[1] - start_s) * 1000)
            ax.set_ylim(0, top)
            ax.grid(not section.spectrogram)
            meta = first.loc[interval_id]
            mae = " · ".join(
                f"{_SHORT_LABELS.get(t, t)} {scores.get((interval_id, t), np.nan):.0f}" for t in trackers
            )
            ax.set_title(f"{meta['vowel']}  {meta['speaker_id']}  {_shorten(item_id)}\nMAE {mae}", loc="left")
        for ax in axes.flat[len(interval_ids):]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel("ms from vowel start")
        for ax in axes[:, 0]:
            ax.set_ylabel("Hz")
        gold = Line2D([], [], color=_TEXT, linewidth=1.6, marker="o" if static else None, markersize=3,
                      linestyle="none" if static else "-", label="Gold")
        fig.legend(handles=[gold, *_tracker_handles(trackers, marker=False)], loc="outside lower center",
                   ncols=len(trackers) + 1)
        suffix = f" (page {page + 1})" if page else ""
        fig.suptitle(
            f"{_dataset_label(dataset.manifest.name)}: {section.title}{suffix}. {section.description}. "
            "Dotted lines mark the vowel; faded lines lie outside it; MAE in Hz.",
            x=0.01, ha="left", fontweight="bold",
        )
    return fig


def _gold_points(dataset, interval_id: str, item_id: str, start_s: float, end_s: float, static: bool) -> pd.DataFrame:
    if static:
        points = dataset.static_measurements.loc[dataset.static_measurements["interval_id"] == interval_id].copy()
        relative = start_s + points["relative_position"] * (end_s - start_s)
        points["time_s"] = points["time_s"].where(points["time_s"].notna(), relative)
        return points.dropna(subset=["time_s"])
    tracks = dataset.tracks
    return tracks.loc[
        (tracks["item_id"] == item_id) & (tracks["time_s"] >= start_s - 1e-9) & (tracks["time_s"] <= end_s + 1e-9)
    ]


def _draw_gold(ax, dataset, interval_id: str, item_id: str, start_s: float, end_s: float, static: bool) -> None:
    points = _gold_points(dataset, interval_id, item_id, start_s, end_s, static)
    times = (points["time_s"].to_numpy(dtype=float) - start_s) * 1000
    for formant in _FORMANTS:
        if static:
            ax.plot(times, points[formant], color=_TEXT, marker="o", markersize=3, linestyle="none", zorder=4)
        else:
            ax.plot(times, points[formant], color=_TEXT, linewidth=1.6, zorder=4)


def _draw_prediction(ax, predictions: pd.DataFrame, item_id: str, window: tuple[float, float], start_s: float,
                     end_s: float, tracker: str) -> None:
    frame = predictions.loc[
        (predictions["item_id"] == item_id)
        & (predictions["time_s"] >= window[0])
        & (predictions["time_s"] <= window[1])
    ].sort_values("time_s")
    seconds = frame["time_s"].to_numpy(dtype=float)
    # Full strength from the last frame before the vowel to the first frame after it:
    # the stretch the evaluator interpolates from. Everything else is faded.
    inside = np.flatnonzero((seconds >= start_s - 1e-9) & (seconds <= end_s + 1e-9))
    before = np.flatnonzero(seconds < start_s - 1e-9)
    after = np.flatnonzero(seconds > end_s + 1e-9)
    if len(inside):
        first = before[-1] if len(before) else inside[0]
        last = after[0] if len(after) else inside[-1]
    else:
        first = last = None
    index = _slot(tracker)
    style = {"color": _series(index), "linestyle": _LINESTYLES[index % 3], "linewidth": 1.1, "zorder": 3}
    times = (seconds - start_s) * 1000
    for formant in _FORMANTS:
        values = frame[formant].to_numpy(dtype=float)
        ax.plot(times, values, alpha=0.3, **style)
        if first is not None:
            ax.plot(times[first:last + 1], values[first:last + 1], **style)


def _top_frequency(dataset, interval_id, item_id, start_s, end_s, static, predictions, window) -> float:
    """Upper y limit: just above the highest gold or predicted F3, but within 1500-6000 Hz."""
    values = [_gold_points(dataset, interval_id, item_id, start_s, end_s, static)["F3"]]
    for frame in predictions.values():
        selected = frame.loc[(frame["item_id"] == item_id) & frame["time_s"].between(window[0], window[1]), "F3"]
        values.append(selected)
    highest = pd.concat(values).max()
    return float(np.clip((highest if np.isfinite(highest) else 4000.0) * 1.15, 1500.0, 6000.0))


def _audio_path(value, root: Path | None) -> Path | None:
    if value is None or pd.isna(value):
        return None
    path = Path(str(value)).expanduser()
    return root / path if not path.is_absolute() and root is not None else path


def _draw_spectrogram(ax, audio_path: Path | None, window: tuple[float, float], start_s: float, top: float) -> None:
    """Wideband spectrogram behind the tracks; skipped quietly if the audio cannot be read."""
    if audio_path is None or not audio_path.is_file():
        return
    try:
        samples, rate = _read_wav(audio_path, *window)
    except (wave.Error, OSError, ValueError):
        return
    length = max(16, round(0.005 * rate))
    hop = max(1, round(0.001 * rate))
    if len(samples) < length:
        return
    emphasised = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    n_fft = int(2 ** np.ceil(np.log2(max(length, 512))))
    starts = np.arange(0, len(emphasised) - length + 1, hop)
    frames = np.stack([emphasised[s:s + length] for s in starts]) * np.hanning(length)
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    decibels = 10 * np.log10(power + 1e-12)
    decibels = np.clip(decibels, decibels.max() - 60, None)
    frequencies = np.fft.rfftfreq(n_fft, 1 / rate)
    keep = frequencies <= top
    centres = (window[0] + (starts + length / 2) / rate - start_s) * 1000
    ax.pcolormesh(centres, frequencies[keep], decibels[:, keep].T, cmap=_SPECTROGRAM, shading="nearest",
                  rasterized=True, zorder=1)


def _read_wav(path: Path, start_s: float, end_s: float) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as reader:
        rate = reader.getframerate()
        width = reader.getsampwidth()
        channels = reader.getnchannels()
        first = max(0, round(start_s * rate))
        last = min(reader.getnframes(), round(end_s * rate))
        reader.setpos(first)
        raw = reader.readframes(max(0, last - first))
    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}
    if width not in dtypes:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    samples = np.frombuffer(raw, dtype=dtypes[width]).astype(float)
    return samples.reshape(-1, channels)[:, 0], rate


def _shorten(item_id: str, limit: int = 28) -> str:
    # Full ids are in the CSV next to the PDF; the panel title only has room for the end.
    return item_id if len(item_id) <= limit else "…" + item_id[-(limit - 1):]
