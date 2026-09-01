# Hillenbrand et al. (1995)

The Hillenbrand adapter prepares the original American English /hVd/ recordings as a **static-gold** benchmark dataset. Each WAV remains one source-native item, while the manually marked vowel nucleus becomes a source `vowel` interval.

## Required source files

Only the following source data are required:

```text
Hillenbrand/
├── bigdata.dat.txt
├── timedata.dat.txt
├── men/
│   └── *.wav
├── women/
│   └── *.wav
└── kids/
    └── *.wav
```

`bigdata.dat.txt` is used instead of `vowdata.dat.txt` because it contains the same static formant information at finer temporal sampling: steady state plus 10%, 20%, ..., 80% of vowel duration.

The benchmark does not currently use `iddata.dat.txt`, `misid.dat.txt`, `vowdata.ds.txt`, or `vowdata.dat.txt`. Those files contain perceptual/descriptive information or a coarser subset of measurements and are not required to reproduce the benchmark gold measurements.

The expected filenames encode speaker group, talker number, and vowel, for example `m12ae.wav`:

- `m`: man
- `w`: woman
- `b`: boy
- `g`: girl
- characters 2-3: talker number
- characters 4-5: vowel code

The adapter preserves `speaker_group` (`man`, `woman`, `boy`, `girl`), `age_group` (`adult`, `child`), binary source-implied gender, vowel code, and the corresponding /hVd/ word as item metadata.

## Static measurements

Each token normally contributes nine independent source-defined static observations:

```text
steady_state
10%
20%
30%
40%
50%
60%
70%
80%
```

The percentage samples are stored with `relative_position` values `0.1` through `0.8` and reference the source vowel interval. They are not converted into a synthetic trajectory.

`timedata.dat.txt` supplies the vowel nucleus `Start`/`End` times and two independent steady-state judgments (`Center1`, `Center2`). `bigdata.dat.txt` exposes only one steady-state F1-F3 tuple, so the adapter applies this deterministic location policy:

1. Use `Center1` when it falls within the inclusive vowel nucleus `[Start, End]`.
2. If `Center1` is outside the vowel nucleus but `Center2` is inside, use `Center2` as a fallback.
3. If neither center lies within the vowel nucleus, omit only the `steady_state` measurement. Keep the item, vowel interval, and all valid 10%-80% measurements.

Each retained steady-state row records `steady_state_time_source` as `center1` or `center2`. Both original judge times are also preserved as metadata.

A source formant value of `0` means that the formant was not measurable. Such values become missing/NaN in the prepared dataset rather than literal 0 Hz measurements.

Hillenbrand `bigdata.dat.txt` contains F1-F3 only, so the prepared dataset declares:

```text
F1 F2 F3
```

F4 remains null.

## Local setup

The example config expects:

```text
data/raw/hillenbrand/Hillenbrand/
```

The current shared Oka copy is under:

```text
/media/share/corpora/Hillenbrand/Hillenbrand/
```

From the repository root, one simple way to copy only the files needed by the adapter is:

```bash
mkdir -p data/raw/hillenbrand/Hillenbrand

rsync -av \
  --include='bigdata.dat.txt' \
  --include='timedata.dat.txt' \
  --include='men/' --include='men/*.wav' \
  --include='women/' --include='women/*.wav' \
  --include='kids/' --include='kids/*.wav' \
  --exclude='*' \
  YOUR_USERNAME@oka:/media/share/corpora/Hillenbrand/Hillenbrand/ \
  data/raw/hillenbrand/Hillenbrand/
```

If your SSH hostname differs from `oka`, replace it accordingly.

## Prepare and inspect

```bash
uv run formant-benchmark dataset prepare \
  --config configs/datasets/hillenbrand.yaml \
  --output prepared/hillenbrand

uv run formant-benchmark dataset inspect prepared/hillenbrand
```

The resulting dataset has `annotation_type: static`, an empty `tracks.parquet`, one vowel interval per item, no invented train/dev/test split, and `static_measurements.parquet` containing the source-defined observations that have interpretable locations.
