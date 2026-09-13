# FastTrackPy environment

## Create the environment

One option with `uv` is:

```bash
uv venv .venvs/fasttrackpy --python 3.12
uv pip install --python .venvs/fasttrackpy/bin/python "fasttrackpy==0.6.1"
```

On Windows, the Python executable is normally:

```text
.venvs/fasttrackpy/Scripts/python.exe
```

The wrapper itself lives in this repository. The isolated interpreter therefore needs
this repository's `src/` directory on `PYTHONPATH`, or an editable no-dependencies
installation of the benchmark package. For example, from the repository root:

```bash
uv pip install --python .venvs/fasttrackpy/bin/python --no-deps -e .
```

`--no-deps` is important: the FastTrackPy environment should contain FastTrackPy's
requirements, not the benchmark's pandas/PyArrow stack.

Then copy/edit `configs/trackers/fasttrackpy.yaml` so `execution.command` points to
that environment's Python executable.

## Verify

```bash
uv run formant-benchmark tracker check fasttrackpy \
  --config configs/trackers/fasttrackpy.yaml
```

The check verifies both the configured runtime and that the wrapper sees exactly
`fasttrackpy==0.6.1`.
