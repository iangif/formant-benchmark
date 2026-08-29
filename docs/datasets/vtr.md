# Preparing VTR locally

The benchmark uses the downloaded **MSR-UCLA VTR-Formant** directory for formant
tracks/segmentations and matching TIMIT audio for the waveforms. VTR is kept at
utterance granularity: one `.fb` file is one benchmark item, while phones, words,
and derived vowels are intervals inside that utterance.

## Required local layout

Place the downloaded VTR directory and the matching TIMIT audio under
`data/raw/vtr/`:

```text
data/raw/vtr/
├── VTRFormants/
│   ├── Train/
│   │   └── dr*/<speaker>/
│   │       ├── *.fb
│   │       ├── *.phn
│   │       └── *.wrd
│   └── Test/
│       └── dr*/<speaker>/
│           ├── *.fb
│           ├── *.phn
│           └── *.wrd
└── TIMIT_fixed/
    ├── TRAIN/
    │   └── DR*/<SPEAKER>/*.WAV
    └── TEST/
        └── DR*/<SPEAKER>/*.WAV
```

The repository ignores `data/`; neither VTR nor TIMIT should be committed.

The VTR archive supplies `.fb`, `.phn`, and `.wrd`. Audio is not duplicated in the
VTR download. For each VTR utterance, the corresponding TIMIT file has the same
split/dialect/speaker/sentence identity and a `.WAV` extension. The TIMIT copy on
oka uses uppercase `TRAIN`/`TEST`, directory names, and filenames.

## Copying only the needed TIMIT audio from oka

The matching corpus is available on oka at:

```text
/media/share/corpora/TIMIT_fixed/
```

After placing VTR at `data/raw/vtr/VTRFormants`, generate an `rsync --files-from`
list containing exactly the utterances present in VTR. The list deliberately
converts the TIMIT side of each path to uppercase:

```bash
mkdir -p data/raw/vtr/TIMIT_fixed

python - <<'PY'
from pathlib import Path

root = Path("data/raw/vtr/VTRFormants")
out = Path("data/raw/vtr/vtr_audio_files.txt")
paths = []
for source_split, timit_split in (("Train", "TRAIN"), ("Test", "TEST")):
    for fb in sorted((root / source_split).glob("dr*/*/*.fb")):
        rel = fb.relative_to(root / source_split).with_suffix(".WAV")
        paths.append(
            (Path(timit_split) / Path(*[part.upper() for part in rel.parts])).as_posix()
        )
out.write_text("\n".join(paths) + "\n", encoding="utf-8")
print(f"wrote {len(paths)} paths to {out}")
PY
```

Then copy those files while preserving the uppercase TIMIT hierarchy:

```bash
rsync -av --files-from=data/raw/vtr/vtr_audio_files.txt \
  USER@oka:/media/share/corpora/TIMIT_fixed/ \
  data/raw/vtr/TIMIT_fixed/
```

Replace `USER` with your oka username. This copies only the TIMIT utterances needed
by VTR rather than the full corpus.

## What the adapter preserves

For each VTR `.fb` utterance, the adapter:

1. keeps the full utterance as one `item_type: utterance` item;
2. records the VTR `Train`/`Test` partition as benchmark `train`/`test` splits;
3. reads speaker sex and dialect from the TIMIT-style directory structure;
4. converts `.phn` sample boundaries to source `phone` intervals;
5. converts `.wrd` sample boundaries to source `word` intervals;
6. derives `vowel` intervals from the configured TIMIT vowel inventory;
7. reads the big-endian HTK `.fb` trajectory and converts kHz to Hz; and
8. preserves all four provided formant trajectories, `F1` through `F4`.

The VTR files contain F1-F4 and B1-B4. The source documentation notes that only
F1-F3 were manually corrected, while F4 remained automatic. Preparation nevertheless
preserves F4 because it is part of the source data. The prepared manifest therefore
declares `F1`, `F2`, `F3`, and `F4` as available. Whether F4 should participate in a
benchmark is an **evaluation-time choice**; users can later select only F1-F3 or any
other compatible non-empty subset.

Bandwidth columns are not part of the benchmark's canonical formant-track schema and
are not imported.

The VTR manual states that successive track vectors are 10 ms apart. Preparation
therefore uses a 0.010 s frame step. This is intentional: the bundled HTK header's
sample-period field is inconsistent with the manual and with utterance duration.

The source warns that interpolation places VTR values in silence regions where no
true VTR exists. The adapter does not acoustically clean these values. Phone/vowel
intervals are retained so later evaluation can explicitly choose the scientifically
appropriate region; `scope=vowels` will avoid silence regions.

The downloaded source also contains a known zero-length word alignment in at least
one `.wrd` file. Zero-length intervals cannot be represented by the benchmark
interval schema, so the adapter skips zero-length **word** intervals and records the
number skipped in `dataset.yaml` as `skipped_zero_length_word_intervals`. Phone
intervals remain strict because they define vowel evaluation regions.

## Vowel inventory

`configs/datasets/vtr.yaml` contains the default TIMIT vowel phone inventory used to
derive vowel intervals. It can be edited explicitly if an experiment needs a
different definition. Syllabic consonants such as `el`, `em`, `en`, and `eng` are
not included by default.

## Prepare and inspect

Install the project environment, then run:

```bash
uv sync --extra dev

uv run formant-benchmark dataset prepare \
  --config configs/datasets/vtr.yaml \
  --output prepared/vtr
```

Inspect the prepared dataset with:

```bash
uv run formant-benchmark dataset inspect prepared/vtr
```

The output should report trajectory data with available formants `F1`, `F2`, `F3`,
and `F4`, source `train`/`test` splits, source phone/word intervals, and derived
vowel intervals. Preparation refuses to overwrite an existing destination unless
`--overwrite` is passed.
