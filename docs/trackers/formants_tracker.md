# FormantsTracker integration

## Benchmark semantics

This integration runs the released MLSpeech FormantsTracker model described in
*Formant Estimation and Tracking using Probabilistic Heat-Maps* (Shrem, Kreuk, and
Keshet, Interspeech 2022).

The upstream implementation predicts three probability heatmaps and converts them to
F1, F2, and F3 trajectories after Gaussian smoothing. The benchmark therefore declares:

```text
F1
F2
F3
```

as the supported formants. F4 is not advertised even though the upstream configuration
contains historical F4-related fields: the released `FormantTracker` model constructs
and returns only F1-F3 decoders.

Supported benchmark input modes are:

```text
full_item
cropped_intervals
```

`full_item_with_intervals` is not advertised because the model does not consume gold
interval boundaries itself. For VTR, `cropped_intervals --interval-type vowel` gives
the model one vowel crop at a time. Generic `--interval-padding` may add acoustic
context, and the benchmark trims persisted predictions back to the requested interval.

## Persistent inference

The benchmark starts one wrapper process for a run. The wrapper loads the model and
checkpoint once, then serves tracking inputs sequentially over the generic streaming
protocol. It does not invoke upstream `main.py` once per item and does not reload the
neural network for every vowel.

If a run intentionally resolves a different checkpoint or device for a later input,
the wrapper reloads the runtime for that new `(checkpoint, device)` pair. Under the
normal configuration, one model instance is reused for the entire run.

## Released preprocessing and output

The wrapper preserves the released inference choices:

```text
model sample rate:       16000 Hz
FFT size:                512
pre-emphasis:            0.97
spectrogram normalization: true
Gaussian kernel size:    7
Gaussian sigma:          1
output step:             10 ms
```

The released checkpoint assumes 16 kHz audio. If an input WAV has another sample rate,
the wrapper resamples a temporary copy to 16 kHz before running the upstream feature
extractor. The prepared/source audio is never modified.

After model inference, the wrapper applies the same Gaussian heatmap smoothing,
argmax-bin conversion, bin-to-Hz mapping, and integer-Hz quantization used by the
released solver. Wrapper-relative 10 ms timestamps are then handled by the generic
benchmark runner, which restores cropped predictions to parent-item coordinates.

## Parameters

Only two tracker parameters are exposed initially:

```text
checkpoint
    path to a compatible FormantsTracker state dict; relative paths are resolved
    from execution.working_directory

device
    auto | cpu | cuda
```

`auto` uses CUDA when available and otherwise CPU. Requesting `cuda` when CUDA is not
available fails clearly.

Architecture/preprocessing fields such as FFT size, network block counts, dropout, and
Gaussian smoothing are intentionally fixed to the released configuration because the
official checkpoint was trained for that model definition. They are not treated as
ordinary benchmark tuning parameters.

## Example runs

Full source-native vowel recordings:

```bash
uv run formant-benchmark track \
  --dataset prepared/hillenbrand \
  --tracker formants_tracker \
  --tracker-config configs/trackers/formants_tracker.yaml \
  --input-mode full_item \
  --output runs/formants-tracker-hillenbrand
```

Vowel crops from VTR with acoustic context:

```bash
uv run formant-benchmark track \
  --dataset prepared/vtr \
  --tracker formants_tracker \
  --tracker-config configs/trackers/formants_tracker.yaml \
  --input-mode cropped_intervals \
  --interval-type vowel \
  --interval-padding 0.035 \
  --output runs/formants-tracker-vtr-vowels
```

The prediction run remains evaluation-independent. `predictions.parquet` contains the
normalized pre-evaluation F1-F3 trajectory; later `all` or `vowels` evaluation uses the
same run without rerunning the model.
