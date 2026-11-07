# formant-benchmark

**A reproducible benchmark for formant trackers.** Run any tracker on any speech dataset that has hand-verified formant measurements, and compare all of them with the same metrics, alignment rules, and reports.

Formant trackers are usually compared on different data, with different settings and with different definitions of "error". This framework puts dataset ingestion, tracker execution, and evaluation behind common interfaces. Adding a new dataset or tracker requires writing one adapter.

Built at the [Montréal Computational & Quantitative Linguistics Lab](https://mcqll.org/). It evaluates trackers on public corpora and on the lab's own gold dataset (English, French, Japanese, Uyghur, ...), which was annotated with [formant-annotation-tool](https://github.com/iangif/formant-annotation-tool).

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python%203.11+-3776AB?logo=python&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas%20%2B%20Parquet-150458?logo=pandas&logoColor=white">
  <img alt="Pydantic" src="https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white">
</p>

## Design highlights

- **Two plug-in points.** *Dataset adapters* turn each source format into one prepared schema: items, intervals, gold tracks, metadata and splits, stored as Parquet with a content fingerprint. *Tracker wrappers* run each tracker in its own environment and exchange data through a small JSON protocol. Everything in between is shared.
- **Expensive work is never repeated.** Each tracking run is keyed by a digest of everything that affects its predictions: dataset fingerprint, tracker version, resolved config, parameter layers, and inputs. Experiments with matching conditions reuse existing predictions, and interrupted runs can resume from checkpoints.
- **Configuration errors fail before tracking starts.** Override rules are checked against real metadata fields before any tracking runs, so a typo like `gendr` stops the run immediately.
- **Clear evaluation definitions.** Predictions are aligned to gold at the frame level, and each metric (MAE, RMSE, formant detection rate and coverage) has one written definition. Results are reported per evaluation unit, so you can break them down by speaker, vowel or gender.
- **Reports.** `formant-benchmark report` produces tracker × dataset heatmaps, confidence intervals bootstrapped over speakers, error ECDFs, error by position in the vowel, and vowel-space plots.

## Datasets and trackers

| Datasets | Gold type | 
|---|---
| [Hillenbrand et al. (1995)](docs/datasets/hillenbrand.md) | static (vowel nucleus + 10–80%) 
| [MSR-UCLA VTR-Formant](docs/datasets/vtr.md) | full trajectories 
| [MCQLL formants](docs/datasets/mcqll.md) | full trajectories, 4+ languages 

| Trackers |
|---|
| [Praat (Burg)](docs/trackers/praat_burg.md) |
| [FastTrackPy](docs/trackers/fasttrackpy.md) |
| [FormantsTracker](docs/trackers/formants_tracker.md) |

## Setup

```bash
uv sync --extra dev
source .venv/bin/activate
pytest                     # 168 tests, using synthetic data fixtures
```

Each tracker runs in its own environment. See setup: [docs/environments](docs/environments)
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
formant-benchmark dataset prepare --config hillenbrand
formant-benchmark experiment run hillenbrand-praat_burg
formant-benchmark report
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
| `formant-benchmark report` | Plot all evaluations into `reports/` (needs `uv sync --extra analysis`) |
| `formant-benchmark examples` | Plot example vowels' gold and predicted tracks into `reports/examples/` |

Add `--help` to any command for its options. For `track`, `evaluate`, tracker
parameters and run outputs, see [docs/trackers/infrastructure.md](docs/trackers/infrastructure.md).
