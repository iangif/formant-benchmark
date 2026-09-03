"""Command-line entry point for dataset preparation and tracker execution."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import yaml

from formant_benchmark.config.loading import load_yaml
from formant_benchmark.data.io import inspect_prepared_dataset, load_prepared_dataset
from formant_benchmark.data.models import TrackingInputMode
from formant_benchmark.datasets import DATASET_REGISTRY, register_builtin_datasets
from formant_benchmark.exceptions import (
    ConfigurationError,
    FormantBenchmarkError,
    TrackerExecutionError,
)
from formant_benchmark.execution.backends import backend_from_config
from formant_benchmark.execution.inputs import build_tracking_inputs
from formant_benchmark.preparation.fingerprint import dataset_fingerprint
from formant_benchmark.preparation.pipeline import prepare_dataset
from formant_benchmark.runs.io import inspect_prediction_run
from formant_benchmark.trackers import TRACKER_REGISTRY, register_builtin_trackers


def build_parser() -> argparse.ArgumentParser:
    """Build dataset and tracker CLI commands."""
    parser = argparse.ArgumentParser(prog="formant-benchmark")
    subcommands = parser.add_subparsers(dest="command", required=True)

    dataset = subcommands.add_parser("dataset", help="Prepare and inspect gold datasets.")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)

    dataset_commands.add_parser("list", help="List available dataset adapters.")

    prepare = dataset_commands.add_parser("prepare", help="Prepare a dataset from YAML configuration.")
    prepare.add_argument("--config", required=True, type=Path, help="Dataset YAML configuration path.")
    prepare.add_argument("--output", required=True, type=Path, help="Prepared dataset output directory.")
    prepare.add_argument("--overwrite", action="store_true", help="Safely replace an existing prepared dataset.")

    inspect = dataset_commands.add_parser("inspect", help="Inspect a prepared dataset directory.")
    inspect.add_argument("dataset", type=Path, help="Prepared dataset directory.")

    tracker = subcommands.add_parser("tracker", help="Inspect and check tracker integrations.")
    tracker_commands = tracker.add_subparsers(dest="tracker_command", required=True)
    tracker_commands.add_parser("list", help="List available tracker adapters.")
    tracker_inspect = tracker_commands.add_parser("inspect", help="Show tracker capabilities.")
    tracker_inspect.add_argument("tracker", help="Registered tracker name.")
    tracker_check = tracker_commands.add_parser("check", help="Check configured execution prerequisites.")
    tracker_check.add_argument("tracker", help="Registered tracker name.")
    tracker_check.add_argument("--config", type=Path, help="Optional tracker YAML configuration.")

    track = subcommands.add_parser("track", help="Create or explicitly resume a PredictionRun.")
    track.add_argument("--dataset", required=True, type=Path, help="Prepared dataset directory.")
    track.add_argument("--tracker", required=True, help="Registered tracker name.")
    track.add_argument("--tracker-config", type=Path, help="Tracker YAML configuration.")
    track.add_argument("--experiment-config", type=Path, help="Experiment parameter/override YAML.")
    track.add_argument("--output", required=True, type=Path, help="Prediction run output directory.")
    track.add_argument(
        "--input-mode",
        required=True,
        choices=[value.value for value in TrackingInputMode],
        help="Audio/information supplied to the tracker.",
    )
    track.add_argument("--interval-type", help="Interval type for cropped or interval-aware input.")
    track.add_argument("--split", help="Track only items in this prepared split.")
    track.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Highest-precedence tracker parameter; repeat as needed.",
    )
    track.add_argument("--resume", action="store_true", help="Explicitly resume a compatible existing run.")
    track.add_argument("--fail-fast", action="store_true", help="Stop after the first item-level failure.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and translate expected application failures into concise errors."""
    parser = build_parser()
    args = parser.parse_args(argv)
    register_builtin_datasets()
    register_builtin_trackers()

    try:
        if args.command == "dataset" and args.dataset_command == "list":
            for name in DATASET_REGISTRY.names():
                print(name)
            return 0

        if args.command == "dataset" and args.dataset_command == "prepare":
            config = load_yaml(args.config)
            adapter_name = config.get("adapter")
            if not isinstance(adapter_name, str) or not adapter_name.strip():
                raise ConfigurationError("Dataset configuration must define a non-empty 'adapter'.")
            adapter_type = DATASET_REGISTRY.get(adapter_name)
            prepared = prepare_dataset(
                adapter_type(),
                config,
                args.output,
                overwrite=args.overwrite,
            )
            summary = inspect_prepared_dataset(prepared)
            print(yaml.safe_dump(summary, sort_keys=False).rstrip())
            return 0

        if args.command == "dataset" and args.dataset_command == "inspect":
            prepared = load_prepared_dataset(args.dataset)
            print(yaml.safe_dump(inspect_prepared_dataset(prepared), sort_keys=False).rstrip())
            return 0

        if args.command == "tracker" and args.tracker_command == "list":
            for name in TRACKER_REGISTRY.names():
                print(name)
            return 0

        if args.command == "tracker" and args.tracker_command == "inspect":
            tracker = TRACKER_REGISTRY.get(args.tracker)()
            capabilities = {
                "name": tracker.name,
                "version": tracker.version,
                "formants": [value.value for value in sorted(tracker.capabilities.formants, key=lambda x: x.value)],
                "input_modes": [value.value for value in sorted(tracker.capabilities.input_modes, key=lambda x: x.value)],
                "interval_types": sorted(tracker.capabilities.interval_types),
                "requires_audio": tracker.capabilities.requires_audio,
            }
            print(yaml.safe_dump(capabilities, sort_keys=False).rstrip())
            return 0

        if args.command == "tracker" and args.tracker_command == "check":
            tracker = TRACKER_REGISTRY.get(args.tracker)()
            config = _tracker_config(args.config, tracker.name)
            effective = _merge_tracker_defaults(tracker.default_configuration, config)
            backend = backend_from_config(effective)
            result = backend.check(tracker.wrapper_command(effective))
            result["tracker"] = tracker.name
            print(yaml.safe_dump(result, sort_keys=False).rstrip())
            if not result["available"]:
                raise TrackerExecutionError("Tracker prerequisites are not available; see check output above.")
            return 0

        if args.command == "track":
            prepared = load_prepared_dataset(args.dataset)
            tracker = TRACKER_REGISTRY.get(args.tracker)()
            tracker_config = _tracker_config(args.tracker_config, tracker.name)
            experiment_config = load_yaml(args.experiment_config) if args.experiment_config else {}
            cli_parameters = _parse_parameters(args.parameter)
            input_mode = TrackingInputMode(args.input_mode)
            with tempfile.TemporaryDirectory(prefix="formant-benchmark-inputs-") as temporary:
                inputs = build_tracking_inputs(
                    prepared,
                    tracker,
                    input_mode=input_mode,
                    interval_type=args.interval_type,
                    split=args.split,
                    temporary_directory=Path(temporary),
                )
                run = tracker.run(
                    inputs,
                    config=tracker_config,
                    destination=args.output,
                    dataset_fingerprint=prepared.manifest.fingerprint or dataset_fingerprint(prepared),
                    dataset_name=prepared.manifest.name,
                    dataset_tracker_config=prepared.manifest.tracker_overrides.get(tracker.name),
                    experiment_config=experiment_config,
                    cli_parameters=cli_parameters,
                    input_mode=input_mode,
                    interval_type=args.interval_type,
                    split=args.split,
                    resume=args.resume,
                    fail_fast=args.fail_fast,
                    show_progress=True,
                )
            print(yaml.safe_dump(inspect_prediction_run(run), sort_keys=False).rstrip())
            return 0
    except FormantBenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error("Unsupported command.")
    return 2


def _tracker_config(path: Path | None, tracker_name: str) -> dict[str, object]:
    config = load_yaml(path) if path else {}
    configured_name = config.get("tracker")
    if configured_name is not None and configured_name != tracker_name:
        raise ConfigurationError(
            f"Tracker config declares '{configured_name}' but CLI selected '{tracker_name}'."
        )
    return config


def _merge_tracker_defaults(defaults: object, config: dict[str, object]) -> dict[str, object]:
    from formant_benchmark.config.resolution import deep_merge

    return deep_merge(defaults if isinstance(defaults, dict) else {}, config)


def _parse_parameters(values: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for entry in values:
        if "=" not in entry:
            raise ConfigurationError(f"CLI parameter must use KEY=VALUE syntax: '{entry}'.")
        key, raw = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigurationError("CLI parameter keys must be non-empty.")
        if key in result:
            raise ConfigurationError(f"CLI parameter '{key}' was provided more than once.")
        result[key] = yaml.safe_load(raw)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
