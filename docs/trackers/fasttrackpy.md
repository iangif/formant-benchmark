# FastTrackPy integration

## Benchmark semantics

For each tracking input, the wrapper calls FastTrackPy's `process_audio_file()` and
uses the automatically selected `CandidateTracks.winner`. The benchmark persists the
winner's **smoothed** trajectory (`winner.smoothed_formants`) rather than the raw Praat
estimates. Wrapper-relative times are normalized by the generic runner, so cropped
interval predictions are restored to parent-item coordinates before persistence.

Supported input modes are:

```text
full_item
cropped_intervals
```

`full_item_with_intervals` is intentionally not advertised for FastTrackPy.
For VTR-style utterance datasets, `cropped_intervals --interval-type vowel` is the
natural vowel-only FastTrackPy condition. For datasets whose source-native item is
already a vowel recording/token, `full_item` is usually appropriate.

Short exact vowel crops can be too brief for FastTrackPy's internal Praat pitch or
intensity analyses. The benchmark therefore supports generic cropped-input context via
`--interval-padding`. A fixed value such as 25 ms supplies the tracker with surrounding
audio while retaining only predictions from the original vowel interval. This changes
the tracking input condition without changing the later vowel evaluation scope.

## Parameters

The adapter exposes the documented FastTrackPy 0.6.1 processing choices:

```text
min_max_formant
max_max_formant
nstep
n_formants
window_length
time_step
pre_emphasis_from
pitch_floor
smoother_method
smoother_order
loss_method
agg_method
heuristics
```

Supported smoother names are `dct_smooth_regression` and `dct_smooth`; loss names are
`lmse` and `mse`; the aggregation method is `agg_sum`.

The pre-specified heuristic names accepted by benchmark configuration are:

```text
f1_max
f4_min
b2_max
b3_max
rhotic
f3_f4_sep
```

Unknown parameters fail before tracking.
`n_formants` is restricted to 1-4 because the benchmark's canonical formant vocabulary
is F1-F4.

## Example runs

Full source-native items:

```bash
uv run formant-benchmark track \
  --dataset prepared/hillenbrand \
  --tracker fasttrackpy \
  --tracker-config configs/trackers/fasttrackpy.yaml \
  --input-mode full_item \
  --output runs/fasttrackpy-hillenbrand
```

Vowel crops from VTR:

```bash
uv run formant-benchmark track \
  --dataset prepared/vtr \
  --tracker fasttrackpy \
  --tracker-config configs/trackers/fasttrackpy.yaml \
  --input-mode cropped_intervals \
  --interval-type vowel \
  --interval-padding 0.035 \
  --output runs/fasttrackpy-vtr-vowels
```

Dataset metadata overrides use the generic precedence machinery. For example, a
research configuration may override ceiling ranges by `gender` without modifying the
FastTrackPy adapter. The fully resolved values are persisted per input in
`item_parameters.parquet`.
