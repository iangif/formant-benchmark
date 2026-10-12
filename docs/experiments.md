# Experiments

An experiment file describes one condition: a prepared dataset, a tracker, how the
tracker is run, and how its predictions are evaluated. One command runs it:

```bash
cp configs/experiments/template.yaml configs/experiments/vtr_praat_burg_vowels.yaml
# edit the copy, then:
uv run formant-benchmark experiment run vtr_praat_burg_vowels
```

The experiment name is the file name unless the file sets `name`. Tracking writes
`runs/<name>` and evaluation writes `evaluations/<name>`. The template lists every
option with its default.

## Reuse instead of rerunning

Tracking is expensive, so before tracking the command computes the run's
tracking-condition digest: a hash of the dataset fingerprint, tracker and version,
the resolved tracker config, dataset and experiment parameter layers, CLI
`--parameter` values, input mode, interval type, padding, split, and the exact
inputs. Then:

| Situation | Result |
|---|---|
| a completed run anywhere in `runs/` has the same digest | reused; no tracking |
| `runs/<name>` exists with a different digest | error; rename the experiment or remove the run |
| the matching run did not finish | error unless `--resume` is given |
| no match | tracked into `runs/<name>` |

Only the `tracking` section affects the digest. To evaluate the same predictions
several ways, write several experiments with identical `tracking` sections: the
first one tracks, the rest reuse its run.

Evaluation is reused when `evaluations/<name>` already has the same settings and
predictions. If the settings changed, the command stops unless `--overwrite` is given.
Requested values are compared, so omitting an option and writing out its default
count as different settings.

Runs made earlier with `formant-benchmark track` are reused too when their
condition matches.

## Options

```text
--parameter KEY=VALUE   highest-precedence tracker parameter (part of the digest)
--resume                continue a matching unfinished run
--overwrite             replace an evaluation that used different settings
--fail-fast             stop after the first item-level tracking failure
```

Everything else is set in the file, so an experiment name always refers to one
recorded condition.

## Quick runs without a file

`track` and `evaluate` remain available for one-off runs. Datasets can be given by
name (`--dataset vtr` means `prepared/vtr`), the tracker config defaults to
`configs/trackers/<tracker>.yaml`, `evaluate` defaults `--dataset` to the run's
dataset, and outputs default to timestamped names such as
`runs/20261011-200501-vtr-praat_burg`.
