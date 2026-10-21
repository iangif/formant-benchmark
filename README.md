# formant-benchmark

Benchmarks formant trackers against gold-standard formant measurements.

## Setup

```bash
uv sync --extra dev
source .venv/bin/activate
```

Each tracker runs in its own environment. See [docs/environments](docs/environments)
([Praat Burg](docs/environments/praat_burg.md),
[FastTrackPy](docs/environments/fasttrackpy.md),
[FormantsTracker](docs/environments/formants_tracker.md)).

Run all commands from the repository root.

## Pipeline

1. **Raw data** goes in `data/raw/`
   ([Hillenbrand](docs/datasets/hillenbrand.md), [MCQLL](docs/datasets/mcqll.md),
   [VTR](docs/datasets/vtr.md)).
2. **Prepare** each dataset once into `prepared/<dataset>`.
3. **Run an experiment**. It tracks into `runs/<experiment>` and evaluates into
   `evaluations/<experiment>`.

```bash
formant-benchmark dataset prepare --config vtr
formant-benchmark experiment run my_experiment
```

## Experiments

An experiment is a YAML file in `configs/experiments/` that names a dataset, a
tracker, how to track and how to evaluate. Start from the template:

```bash
cp configs/experiments/template.yaml configs/experiments/my_experiment.yaml
formant-benchmark experiment run my_experiment
```

Experiments with the same tracking settings reuse existing predictions instead of
tracking again. See [docs/experiments.md](docs/experiments.md).

## Commands

| Command | What it does |
|---|---|
| `formant-benchmark dataset list` | List dataset adapters |
| `formant-benchmark dataset prepare --config <name>` | Prepare `configs/datasets/<name>.yaml` into `prepared/<name>` |
| `formant-benchmark dataset inspect <name>` | Summarise a prepared dataset |
| `formant-benchmark tracker list` | List trackers |
| `formant-benchmark tracker inspect <tracker>` | Show a tracker's capabilities |
| `formant-benchmark tracker check <tracker>` | Check that a tracker's environment works |
| `formant-benchmark experiment run <name>` | Track and evaluate one experiment |
| `formant-benchmark track ...` | One-off tracking run without an experiment file |
| `formant-benchmark evaluate ...` | One-off evaluation of a run |

Add `--help` to any command for its options. For `track`, `evaluate`, tracker
parameters and run outputs, see [docs/trackers/infrastructure.md](docs/trackers/infrastructure.md).

## Trackers

- [Praat Burg](docs/trackers/praat_burg.md)
- [FastTrackPy](docs/trackers/fasttrackpy.md)
- [FormantsTracker](docs/trackers/formants_tracker.md)

## Tests

```bash
pytest
```
