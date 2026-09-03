# Tracker infrastructure

## Commands

Discover and inspect registered trackers:

```bash
formant-benchmark tracker list
formant-benchmark tracker inspect synthetic
formant-benchmark tracker check synthetic --config configs/trackers/synthetic.yaml
```

`tracker check` verifies the executable/runtime and working directory. It does not
install dependencies or run tracking.

Create a synthetic prediction run from a prepared dataset:

```bash
formant-benchmark track \
  --dataset prepared/vtr \
  --tracker synthetic \
  --tracker-config configs/trackers/synthetic.yaml \
  --input-mode full_item \
  --output runs/synthetic-vtr
```

Track only vowel crops from one prepared split:

```bash
formant-benchmark track \
  --dataset prepared/vtr \
  --tracker synthetic \
  --input-mode cropped_intervals \
  --interval-type vowel \
  --split test \
  --parameter frame_step_s=0.01 \
  --output runs/synthetic-vtr-vowels
```

`cropped_intervals` requires real WAV files even for the synthetic tracker because
the benchmark exercises and verifies the crop operation. Wrapper-relative times are
converted back to source-item coordinates before persistence. `voiced` interval input
fails explicitly because pitch-based voiced segmentation is deferred.

Resume is never inferred. Use `--resume` with the same dataset fingerprint, tracker,
configuration, input mode, interval type, split, and selected input units:

```bash
formant-benchmark track ... --output runs/synthetic-vtr --resume
```

An incompatible resume fails instead of mixing predictions from different
experimental conditions.

## Configuration precedence

Parameters resolve independently for every tracking input in this order:

1. tracker built-in defaults;
2. tracker YAML;
3. the selected tracker's entry under a dataset config's `tracker_overrides`;
4. experiment YAML;
5. repeated CLI `--parameter KEY=VALUE` values.

Metadata rules use equality matching and may reference item or selected-interval
metadata. Matching rules apply from least to most specific. Conflicting rules of the
same specificity fail before tracker execution. Fully resolved parameters are stored
in `item_parameters.parquet`.

Example dataset-side configuration:

```yaml
tracker_overrides:
  synthetic:
    overrides:
      - where:
          gender: female
        parameters:
          offset_hz: 100
```

## Prediction run artifacts

Each run contains:

```text
run_manifest.yaml
predictions.parquet
failures.parquet
item_parameters.parquet
```

`predictions.parquet` contains only normalized pre-evaluation columns:

```text
item_id, time_s, F1, F2, F3, F4
```

All formant columns are nullable. Missing formants and missing frames remain valid
partial predictions. Empty, unreadable, missing, timed-out, and crashed wrapper
outputs become structured failure rows. Runs continue after input-level failures by
default. Results accumulate in memory and the complete artifacts are atomically
checkpointed every 50 attempted inputs by default, as well as on normal completion
and a handled interruption.

## Wrapper protocol

The benchmark starts the configured wrapper once with:

```text
--stream
```

The benchmark and wrapper then exchange one compact JSON object per line over
stdin/stdout. Each request describes one input unit, its audio path (or `null` when
the tracker does not require audio), duration, metadata, intervals, and resolved
parameters. A successful response includes prediction rows with `time_s` and any
supported subset of `F1` through `F4`; an item-level failure response contains its
message. Diagnostic logging belongs on stderr so it cannot corrupt protocol output.
The benchmark validates and normalizes every response before adding it to the run.

One wrapper normally serves the entire run, avoiding repeated environment/model
startup. An ordinary item failure does not stop it. If the process crashes or an
input exceeds `timeout_s`, the benchmark records only that input as failed, starts a
fresh worker, and continues. Local execution supports an explicit command, working
directory, environment, and per-input timeout. Docker or Apptainer starts one
interactive container with a stable work directory mounted at `/work`.

Operational execution settings may be placed in tracker YAML:

```yaml
execution:
  timeout_s: 30
  checkpoint_every: 50
  # Optional safeguard for wrappers that accumulate memory or global state.
  max_items_per_process: 500
```

Omitting `max_items_per_process` keeps one healthy process for the entire run.
These settings are execution controls; resolved scientific tracker parameters remain
stored separately per input in `item_parameters.parquet`.
