# FormantsTracker environment

FormantsTracker is maintained as a separate external repository and environment. The
benchmark does not vendor its network code or checkpoint.

## Clone the upstream implementation

From the `formant-benchmark` repository root:

```bash
git clone https://github.com/MLSpeech/FormantsTracker.git external/FormantsTracker
```

The upstream repository includes the released checkpoint at:

```text
saved_ckpts/196_1125.ckpt
```

The benchmark configuration may point at another compatible checkpoint, but the
released checkpoint is the default.

## Create the isolated environment

The upstream README recommends Python 3.9 and its `requirements.txt` pins the inference
stack, including `torch==1.13.1` and `torchaudio==0.13.1`.

Using `uv`, from the benchmark repository root:

```bash
uv venv .venvs/formants_tracker --python 3.9
uv pip install --python .venvs/formants_tracker -r external/FormantsTracker/requirements.txt
```

Do **not** install the benchmark's normal dependency set into this environment. The
tracker process only needs the lightweight wrapper modules from this repository.

The default tracker config already uses repository-relative paths:

```yaml
execution:
  command: .venvs/formants_tracker/bin/python
  working_directory: external/FormantsTracker
  environment:
    PYTHONPATH: src
```

On Windows, only the virtual-environment executable path changes:

```yaml
command: .venvs/formants_tracker/Scripts/python.exe
```

## Verify

```bash
uv run formant-benchmark tracker check formants_tracker \
  --config configs/trackers/formants_tracker.yaml
```

For a local backend, the check verifies:

- the configured interpreter and upstream working directory exist;
- upstream `model.py`, `dataloader.py`, and `utils.py` import successfully;
- `torch==1.13.1` and `torchaudio==0.13.1` are visible;
- the configured checkpoint exists;
- a requested CUDA device is actually available;
- the upstream git commit is reported when Git metadata is available.

The checkpoint is loaded only when tracking begins, not during `tracker check`.
