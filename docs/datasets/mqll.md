# Preparing MCQLL Formants locally

## Required local source files

For each corpus/language, keep the gold exports and source audio separate:

```text
data/raw/mcqll/<corpus>/
├── gold_tracks/
│   ├── batch1/
│   │   ├── tracks.parquet
│   │   ├── tokens.parquet
│   │   └── export_manifest.json
│   └── batch2/
│       └── ...
└── source/
    ├── batch1/
    │   └── audio/
    │       └── *.wav
    └── batch2/
        └── audio/
            └── *.wav
```

The current example configurations use:

```text
English:  corpus = ls_eng
Japanese: corpus = gp_jpn
```

## Copying from oka

Under the current formants-export layout, the batch gold exports live at:

```text
/projects/xling-measures/export/gold_tracks/<corpus>/<batch>/
```

and the corresponding token audio lives under:

```text
/projects/xling-measures/data/<corpus>/<batch>/audio/
```

A convenient approach is to copy the corpus-level trees while preserving batch
names. For example, from the benchmark repository root:

```bash
mkdir -p data/raw/mcqll/ls_eng/gold_tracks data/raw/mcqll/ls_eng/source

rsync -av USER@oka:/projects/xling-measures/export/gold_tracks/ls_eng/ \
  data/raw/mcqll/ls_eng/gold_tracks/

# To load all batches
rsync -av --include='*/' --include='*.wav' --exclude='*' \
  USER@oka:/projects/xling-measures/data/ls_eng/ \
  data/raw/mcqll/ls_eng/source/

# To load only a few batches
rsync -avm --include='/batch1/' --include='/batch2/' \
  --include='/batch1/**/' --include='/batch2/**/' \
  --include='/batch1/**.wav' --include='/batch2/**.wav' \
  --exclude='*' \
  USER@oka:/projects/xling-measures/data/ls_eng/ data/raw/mcqll/ls_eng/source/  
```

Repeat with `gp_jpn` for Japanese. Replace `USER` with your oka username.

The repository ignores `data/`, so data should remain local and
must not be committed.

## What the adapter does

For every selected batch, the adapter:

1. validates `export_manifest.json` against the configured corpus and batch;
2. keeps only tokens whose final export status is `exported`;
3. locates `<file_stem>.wav` in the matching batch audio directory;
4. treats each exported vowel token as one source-native benchmark item;
5. maps the exporter's raw `F1`-`F4` columns directly to canonical benchmark
   `F1`-`F4` gold tracks; the smoothed `F1_s`-`F4_s` columns are not used as gold;
6. converts timestamps and phone boundaries to item-relative seconds when needed;
7. preserves a source phone interval and adds the corresponding derived vowel
   interval;
8. preserves `batch` as item metadata; and
9. leaves `splits.parquet` empty because annotation batches are not experimental
   train/dev/test splits.

Available formants are inferred from the non-missing raw gold values across
the selected batches, so F4 remains optional.

## Selecting batches

The example configs use:

```yaml
batches: all
```

To prepare only selected batches, replace that with, for example:

```yaml
batches:
  - batch1
  - batch2
```

Batch selection does not create benchmark splits.

## Prepare and inspect

After installing the project environment:

```bash
uv sync --extra dev
```

prepare English with:

```bash
uv run formant-benchmark dataset prepare \
  --config configs/datasets/mcqll_english.yaml \
  --output prepared/mcqll_english
```

or Japanese with:

```bash
uv run formant-benchmark dataset prepare \
  --config configs/datasets/mcqll_japanese.yaml \
  --output prepared/mcqll_japanese
```

Inspect a prepared dataset with:

```bash
uv run formant-benchmark dataset inspect prepared/mcqll_english
```

Preparation refuses to overwrite an existing destination unless `--overwrite`
is passed.
