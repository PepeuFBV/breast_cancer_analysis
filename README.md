# Breast Cancer Analysis

Reproducible mammogram classification pipeline for the INbreast dataset. The
project is organized around reusable Python modules in `pipeline/` and a small
set of entrypoints for preprocessing, training, orchestration, and evaluation.

## Start Here

- Quick start: [`docs/quickstart.md`](docs/quickstart.md)
- Setup details: [`docs/setup.md`](docs/setup.md)
- Testing and smoke checks: [`docs/testing.md`](docs/testing.md)
- GPU validation: [`docs/gpu.md`](docs/gpu.md)
- Troubleshooting: [`docs/troubleshooting.md`](docs/troubleshooting.md)
- Main config: [`configs/experiment.default.json`](configs/experiment.default.json)
- Local dashboard: [`experiment_dashboard.py`](experiment_dashboard.py)
- Paper source: [`article/main.tex`](article/main.tex)

## Main Workflow

1. Put the INbreast dataset under `data/INbreast Release 1.0/`.
2. Create a virtual environment and install dependencies.
3. Run `python preprocess.py`.
4. Run the experiment queue with `python run_experiments.py run` or use the dashboard.
5. Generate the consolidated report with `python evaluate.py`.

## Minimal Setup

Install the OS packages first. On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip build-essential python3-dev
```

Then create the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Supported Python range: 3.10, 3.11, or 3.12.

For development checks:

```bash
python -m pip install -r requirements-dev.txt
python scripts/check_environment.py --require-venv
python scripts/validate_dataset.py
python scripts/check_gpu.py
python scripts/smoke_run.py
python -m ruff check .
python -m black --check .
python -m pytest
```

Get the full dataset here:

![Download button for the INbreast dataset](imgs/breast-cancer-kaggle.png)

## Expected Dataset layout

```text
data/INbreast Release 1.0/
  INbreast.csv
  AllDICOMs/*.dcm
```

## Key Outputs

- Processed images and splits: `artifacts/processed/`
- Training history and predictions: `artifacts/runs/`
- Iterative runner state, logs, and summary: `artifacts/experiments/`
- Final evaluation report: `artifacts/reports/final_comprehensive_results.csv`

The quick start guide has the exact commands for running, stopping, resuming,
and resetting the experiment queue. The default experiment grid can take hours;
use the smoke checks before launching the full battery.
