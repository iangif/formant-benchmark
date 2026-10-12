# Praat Burg integration

## Benchmark semantics

This integration runs exactly one Praat procedure: **Sound: To Formant (burg)...**,
through `parselmouth.Sound.to_formant_burg()` from `praat-parselmouth==0.4.7`
(embedded Praat 6.1.38). No Praat post-processing such as `Formant: Track...` is
applied; other Praat procedures would be separate tracker integrations.

For each tracking input the wrapper loads the WAV with Parselmouth, mixes
multichannel audio to mono, runs Burg, and emits one row per Praat analysis frame at
the frame centre time (`Formant.xs()`). Values are read exactly at those frame
centres, so they are the frame's own estimates rather than interpolations between
frames.

Praat usually finds up to `max_number_of_formants` formants per frame (5 by default).
The benchmark declares and emits:

```text
F1
F2
F3
F4
```

F5 and above are dropped because the canonical vocabulary stops at F4. Frames in
which Praat found fewer formants keep the missing higher formants as missing values;
this is a valid partial prediction, not a failure. An input for which Praat found no
formant in any frame (for example silence) becomes an `invalid_output` failure.

Supported input modes are:

```text
full_item
cropped_intervals
```

`full_item_with_intervals` is not advertised because Burg does not consume interval
boundaries. Praat shrinks its analysis to fit very short sounds, so even very short
vowel crops still return at least one frame. Frames near the edges of short crops are
nevertheless analysed from little signal; `--interval-padding` supplies surrounding
acoustic context while persisted predictions are trimmed back to the target interval.

## Parameters

The parameters are those of Praat's dialog, with Praat's defaults:

```text
time_step               null    (Praat automatic: 25% of window_length)
max_number_of_formants  5.0     (multiple of 0.5; Burg fits 2x this many poles)
maximum_formant         5500.0  Hz
window_length           0.025   s (Praat's Gaussian window is twice this long)
pre_emphasis_from       50.0    Hz
```

Unknown parameters fail before tracking.

The built-in defaults deliberately contain no speaker-dependent logic. Praat's manual
recommends a ceiling of about 5000 Hz for adult male and 5500 Hz for adult female
voices, and higher for children. Set such ceilings through dataset `tracker_overrides`
or experiment YAML, for example:

```yaml
tracker_overrides:
  praat_burg:
    overrides:
      - where:
          gender: male
        parameters:
          maximum_formant: 5000.0
```

The fully resolved values are persisted per input in `item_parameters.parquet`.

## Example runs

Full source-native items:

```bash
uv run formant-benchmark track \
  --dataset prepared/hillenbrand \
  --tracker praat_burg \
  --tracker-config configs/trackers/praat_burg.yaml \
  --input-mode full_item \
  --output runs/praat-burg-hillenbrand
```

Vowel crops from VTR:

```bash
uv run formant-benchmark track \
  --dataset prepared/vtr \
  --tracker praat_burg \
  --tracker-config configs/trackers/praat_burg.yaml \
  --input-mode cropped_intervals \
  --interval-type vowel \
  --interval-padding 0.035 \
  --output runs/praat-burg-vtr-vowels
```
