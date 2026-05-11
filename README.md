# Breast Cancer Analysis

Reusable mammogram classification pipeline for the INbreast dataset. The repository provides preprocessing, experiment orchestration, training, and evaluation entrypoints backed by JSON configuration.

## Purpose

This project prepares INbreast DICOM data, runs configurable training grids across preprocessing and model variants, and generates consolidated evaluation reports.

## Minimal workflow

Assume `.venv` is activated.

```bash
python scripts/bootstrap_env.py --gpu auto
python preprocess.py
python run_experiments.py launch
python run_experiments.py status
python evaluate.py
```

## Primary docs

- Documentation hub: [docs/index.md](docs/index.md)
- Setup: [docs/setup.md](docs/setup.md)
- Quickstart: [docs/quickstart.md](docs/quickstart.md)
- Execution (`run_experiments.py`): [docs/execution.md](docs/execution.md)
- Configuration (`configs/*.json`): [docs/configuration.md](docs/configuration.md)
- GPU/runtime policy: [docs/gpu.md](docs/gpu.md)
- Testing and validation: [docs/testing.md](docs/testing.md)
- Outputs and run artifacts: [docs/artifacts.md](docs/artifacts.md)
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md)
- Repo architecture: [docs/architecture.md](docs/architecture.md)
