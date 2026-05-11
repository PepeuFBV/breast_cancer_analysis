# Setup

This document covers environment bootstrap and readiness checks for this repository. It is setup-only and does not cover experiment orchestration details.

## Purpose

Establish a validated local environment, confirm dataset placement, and verify runtime readiness before preprocessing or training.

## Read this when

- You are setting up this repository on a new machine.
- You need to revalidate an existing environment after dependency or driver changes.

## Source of truth

- `pyproject.toml`
- `scripts/bootstrap_env.py`
- `scripts/check_environment.py`
- `scripts/validate_dataset.py`
- `scripts/check_gpu.py`
- `scripts/check_runtime.py`
- `configs/experiment.default.json`

## Python and virtual environment

Supported Python range is `>=3.10,<3.13`.

```bash
python -m venv .venv
```

Activate before running commands in this documentation set:

- Linux/macOS: `source .venv/bin/activate`
- PowerShell: `.venv\Scripts\Activate.ps1`

## Bootstrap dependencies

Default (GPU optional, CPU fallback allowed):

```bash
python scripts/bootstrap_env.py --gpu auto
```

Require GPU during bootstrap validation:

```bash
python scripts/bootstrap_env.py --gpu required
```

Force CPU-only stack:

```bash
python scripts/bootstrap_env.py --gpu off
```

Install development dependencies too:

```bash
python scripts/bootstrap_env.py --gpu auto --dev
```

## Dataset placement

Expected layout:

```text
data/INbreast Release 1.0/
  INbreast.csv
  AllDICOMs/*.dcm
```

Validate dataset layout:

```bash
python scripts/validate_dataset.py
```

## Readiness checks

Environment and writable artifact paths:

```bash
python scripts/check_environment.py --require-venv
```

Runtime and device checks:

```bash
python scripts/check_gpu.py
python scripts/check_runtime.py --device auto
```

If dataset is not present yet:

```bash
python scripts/check_environment.py --require-venv --skip-dataset
```

## Related docs

- First successful run paths: [quickstart.md](quickstart.md)
- Runner commands and lifecycle: [execution.md](execution.md)
- GPU-specific setup: [gpu.md](gpu.md)
- Troubleshooting setup failures: [troubleshooting.md](troubleshooting.md)
