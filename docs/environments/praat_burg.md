# Praat Burg environment

## Create the environment

One option with `uv` is:

```bash
uv venv .venvs/praat_burg --python 3.12
uv pip install --python .venvs/praat_burg/bin/python "praat-parselmouth==0.4.7"
```

On Windows, the Python executable is normally:

```text
.venvs/praat_burg/Scripts/python.exe
```

The wrapper itself lives in this repository. The isolated interpreter therefore needs
this repository's `src/` directory on `PYTHONPATH`, or an editable no-dependencies
installation of the benchmark package. For example, from the repository root:

```bash
uv pip install --python .venvs/praat_burg/bin/python --no-deps -e .
```

`--no-deps` is important: the Praat Burg environment should contain Parselmouth and
its NumPy dependency, not the benchmark's pandas/PyArrow stack.

Then copy/edit `configs/trackers/praat_burg.yaml` so `execution.command` points to
that environment's Python executable.

## Praat version

Parselmouth embeds a fixed Praat build rather than calling an installed Praat.
`praat-parselmouth==0.4.7` contains **Praat 6.1.38** (January 2021). This is the same
Parselmouth release, and therefore the same Burg implementation, that FastTrackPy 0.6.1
uses internally.

## Verify

```bash
uv run formant-benchmark tracker check praat_burg \
  --config configs/trackers/praat_burg.yaml
```

The check verifies the configured runtime, that the wrapper sees exactly
`praat-parselmouth==0.4.7`, and reports the embedded Praat version.
