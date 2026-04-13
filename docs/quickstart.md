# Quick Start

This guide covers the fastest way to run the full project from raw INbreast
files to the final report.

## 1. Prerequisites

- Python 3.10+
- The INbreast dataset available locally
- Optional: NVIDIA GPU with TensorFlow-compatible drivers

Expected dataset layout:

```text
data/INbreast Release 1.0/
  INbreast.csv
  AllDICOMs/*.dcm
```

## 2. Set Up the Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you want the lint and test tools too:

```bash
python -m pip install -r requirements-dev.txt
```

## 3. Review the Experiment Grid

The default grid lives in
[`configs/experiment.default.json`](../configs/experiment.default.json).

That file controls:

- dataset and artifact paths
- preprocessing combinations
- models to run
- training settings such as folds, epochs, and batch size
- evaluation settings

If the default battery is too large for your machine or time budget, trim the
config before launching the full queue.

## 4. Preprocess the Dataset

```bash
python preprocess.py
```

Main outputs:

- `artifacts/processed/images/train/`
- `artifacts/processed/images/test/`
- `artifacts/processed/splits/train_split.csv`
- `artifacts/processed/splits/test_split.csv`
- `artifacts/processed/splits/split_summary.json`

## 5. Run the Experiment Queue

Foreground run:

```bash
python run_experiments.py run
```

Background run:

```bash
python run_experiments.py launch
```

Local dashboard:

```bash
streamlit run experiment_dashboard.py
```

The dashboard and CLI share the same persisted runner state.

## 6. Check Status, Stop, and Resume

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

## 7. Generate the Final Report

```bash
python evaluate.py
```

Main outputs:

- `artifacts/reports/final_comprehensive_results.csv`
- `artifacts/reports/evaluation_details/`

## 8. Where Everything Is Saved

- Processed data: `artifacts/processed/`
- Training history: `artifacts/runs/history/`
- Predictions: `artifacts/runs/predictions/`
- Runner state: `artifacts/experiments/state/runner_state.json`
- Runner summary: `artifacts/experiments/summary/experiment_runs.csv`
- Runner log: `artifacts/experiments/logs/iterative-runner.log`

## 9. Small Smoke Run

If you want to validate the pipeline before launching the full battery:

```bash
python run_experiments.py run \
  --models "custom cnn" \
  --preprocessing none \
  --no-combined-preprocessing \
  --folds 0 \
  --epochs 3
```

This keeps the same orchestration flow, but with a much smaller queue.
