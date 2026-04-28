# Quick Start

This is the shortest path from a local INbreast copy to a validated run. The
full experiment grid can take hours, so run the checks and smoke command first.

## 1. Prerequisites

- Python 3.10, 3.11, or 3.12
- `python3`, `python3-venv`, `python3-pip`
- Build tools such as `build-essential` and `python3-dev`
- The INbreast dataset available locally
- Optional NVIDIA GPU support; see [`gpu.md`](gpu.md)

Expected dataset layout:

```text
data/INbreast Release 1.0/
  INbreast.csv
  AllDICOMs/*.dcm
```

## 2. Set Up the Environment

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip build-essential python3-dev
```

Create and install:

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
```

## 3. Validate Setup

```bash
python scripts/check_environment.py --require-venv
python scripts/validate_dataset.py
python scripts/check_gpu.py
python scripts/smoke_run.py
```

Use `python scripts/check_gpu.py --require-gpu` only when the run must use GPU.

## 4. Review the Experiment Grid

The default grid lives in
[`configs/experiment.default.json`](../configs/experiment.default.json). It
controls paths, preprocessing combinations, model names, folds, epochs, batch
size, and evaluation settings.

## 5. Preprocess the Dataset

```bash
python preprocess.py
```

Main outputs:

- `artifacts/processed/images/train/`
- `artifacts/processed/images/test/`
- `artifacts/processed/splits/train_split.csv`
- `artifacts/processed/splits/test_split.csv`
- `artifacts/processed/splits/split_summary.json`

## 6. Optional Real Training Smoke

After preprocessing, run a single real training task before the full grid:

```bash
python run_experiments.py run \
  --models "custom cnn" \
  --preprocessing none \
  --no-combined-preprocessing \
  --folds 0 \
  --epochs 1 \
  --limit 1
```

## 7. Run the Full Experiment Queue

Only start this when the checks above pass and you are ready for a long run.

```bash
python run_experiments.py run
```

Local dashboard:

```bash
streamlit run experiment_dashboard.py
```

The dashboard and CLI share the same persisted runner state.

## 8. Check Status, Stop, and Resume

Check progress:

```bash
python run_experiments.py status
```

Request a safe stop:

```bash
python run_experiments.py stop
```

Behavior:

- the current experiment is allowed to finish
- completed results stay saved
- the queue stops at the next safe boundary

Resume later:

```bash
python run_experiments.py run
```

Useful rerun options:

```bash
python run_experiments.py run --rerun-failed
python run_experiments.py run --rerun-completed
```

Reset only the orchestration state:

```bash
python run_experiments.py reset
```

Reset state and remove saved run outputs:

```bash
python run_experiments.py reset --purge-results
```

## 9. Generate the Final Report

```bash
python evaluate.py
```

Main outputs:

- `artifacts/reports/final_comprehensive_results.csv`
- `artifacts/reports/evaluation_details/`

## 10. Where Everything Is Saved

- Processed data: `artifacts/processed/`
- Training history: `artifacts/runs/history/`
- Predictions: `artifacts/runs/predictions/`
- Runner state: `artifacts/experiments/state/runner_state.json`
- Runner summary: `artifacts/experiments/summary/experiment_runs.csv`
- Runner log: `artifacts/experiments/logs/iterative-runner.log`
