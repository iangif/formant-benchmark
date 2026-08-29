"""Command-line entry point for dataset preparation and inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import yaml

from formant_benchmark.config.loading import load_yaml
from formant_benchmark.data.io import inspect_prepared_dataset, load_prepared_dataset
from formant_benchmark.datasets import DATASET_REGISTRY, register_builtin_datasets
from formant_benchmark.exceptions import ConfigurationError, FormantBenchmarkError
from formant_benchmark.preparation.pipeline import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build the V1 CLI parser; later phases add tracker/track/evaluate groups."""
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and translate expected application failures into concise errors."""
    parser = build_parser()
    args = parser.parse_args(argv)
    register_builtin_datasets()

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
    except FormantBenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error("Unsupported command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
