# Testing

This document groups validation commands by speed and purpose. Commands assume `.venv` is activated.

## Purpose

Provide exact command sets for fast checks, pytest markers, smoke validation, and long-run validation.

## Read this when

- You want confidence before large experiment launches.
- You need the default pytest behavior and marker-specific runs.

## Source of truth

- `pyproject.toml`
- `scripts/check_environment.py`
- `scripts/check_runtime.py`
- `scripts/check_gpu.py`
- `scripts/smoke_run.py`
- `scripts/validate_long_runner.py`
- `tests/`

## Fast validation

```bash
python scripts/check_environment.py --require-venv
python scripts/check_runtime.py --device auto
python -m pytest
```

## Pytest markers

Default `python -m pytest` behavior is defined by `pyproject.toml`:

- includes tests under `tests/`
- excludes markers `slow`, `integration`, and `gpu` by default

Useful marker runs:

```bash
python -m pytest -m unit
python -m pytest -m smoke
python -m pytest -m gpu
python -m pytest -m integration -o addopts=""
python -m pytest -m "not slow and not integration and not gpu"
```

## Smoke validation

Synthetic end-to-end smoke path:

```bash
python scripts/smoke_run.py
```

Smoke run with explicit config/artifacts root:

```bash
python scripts/smoke_run.py --config configs/experiment.smoke.json --artifacts-dir artifacts-smoke
```

## Long-run validation

Tiny real-training runner validation:

```bash
python scripts/validate_long_runner.py --combinations 20 --device cpu
```

If GPU availability should be enforced:

```bash
python scripts/validate_long_runner.py --combinations 20 --device gpu --require-gpu
```

## Related docs

- Setup and readiness prerequisites: [setup.md](setup.md)
- Runner lifecycle commands: [execution.md](execution.md)
- GPU/runtime setup: [gpu.md](gpu.md)
- Failure diagnosis paths: [troubleshooting.md](troubleshooting.md)
