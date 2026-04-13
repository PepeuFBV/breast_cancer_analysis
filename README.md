# Breast Cancer Analysis

Reproducible mammogram classification pipeline for the INbreast dataset. The
project is organized around reusable Python modules in `pipeline/` and a small
set of entrypoints for preprocessing, training, orchestration, and evaluation.

## Start Here

- Full setup and end-to-end run: [`docs/quickstart.md`](docs/quickstart.md)
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

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For development checks:

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m black --check .
python -m pytest
```

## Expected Dataset Layout

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
and resetting the experiment queue.
