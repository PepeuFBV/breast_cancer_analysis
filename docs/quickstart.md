# Quick Start

This is the shortest path from a local INbreast copy to a validated run. The
full experiment grid can take hours, so run the checks and smoke command first.

## Unattended Setup (Recommended)

For a fully automated setup that handles everything:

```bash
python3 scripts/unattended_setup.py --gpu auto
```

This will:
1. Bootstrap the Python environment
2. Validate the dataset
3. Run smoke tests
4. Preprocess the data

After this completes successfully, skip to step 7 to run experiments.

For CPU-only mode:

```bash
python3 scripts/unattended_setup.py --gpu off
```

Verify readiness:

```bash
python3 scripts/check_readiness.py
```

## Manual Setup

If you prefer step-by-step control, follow the sections below.

## 1. Prerequisites

- Python 3.10, 3.11, or 3.12
- `python3`, `python3-venv`, `python3-pip`
- Build tools such as `build-essential` and `python3-dev`
- The INbreast dataset available locally
- Optional NVIDIA GPU support; see [`gpu.md`](gpu.md) and [`wsl_gpu_setup.md`](wsl_gpu_setup.md)

Note: GPU is optional. The pipeline runs on CPU if GPU is not available.

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
python3 scripts/bootstrap_env.py --gpu auto
```

For development checks:

```bash
python3 scripts/bootstrap_env.py --gpu auto --dev
```

Use `--gpu required` instead of `--gpu auto` when the run must fail unless
TensorFlow can use the GPU. Use `--gpu off` to force CPU-only mode.

Note: If GPU setup fails with `--gpu auto`, the bootstrap will succeed and the
pipeline will run on CPU. For WSL GPU troubleshooting, see
[`wsl_gpu_setup.md`](wsl_gpu_setup.md).

## 3. Validate Setup

```bash
./.venv/bin/python scripts/check_environment.py --require-venv
./.venv/bin/python scripts/validate_dataset.py
./.venv/bin/python scripts/check_gpu.py
./.venv/bin/python scripts/smoke_run.py
```

Use `./.venv/bin/python scripts/check_gpu.py --require-gpu` only when the run
must use GPU.

Validate runtime device policy explicitly:

```bash
./.venv/bin/python scripts/check_runtime.py --device cpu
./.venv/bin/python scripts/check_runtime.py --device auto
./.venv/bin/python scripts/check_runtime.py --device gpu || true
```

These checks are the default validation path. Do not start with the full
experiment grid.

## 4. Review the Experiment Grid

The default grid lives in
[`configs/experiment.default.json`](../configs/experiment.default.json). It
controls paths, preprocessing combinations, model names, folds, epochs, batch
size, and evaluation settings.

The shipped config disables pairwise preprocessing combinations by default.
Enable them only when you intentionally want a much larger queue:

```bash
./.venv/bin/python run_experiments.py run --combined-preprocessing
```

## 5. Preprocess the Dataset

```bash
./.venv/bin/python preprocess.py
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
./.venv/bin/python run_experiments.py run \
  --models "custom cnn" \
  --preprocessing none \
  --no-combined-preprocessing \
  --folds 0 \
  --epochs 1 \
  --limit 1
```

Long-run orchestration smoke (recommended before full queue):

```bash
./.venv/bin/python scripts/validate_long_runner.py --combinations 20 --device cpu
./.venv/bin/python scripts/validate_long_runner.py --combinations 20 --device auto
```

If the runner later fails after several combinations, inspect:

- `artifacts/experiments/logs/iterative-runner.log`
- `artifacts/experiments/logs/run-events.jsonl`
- `artifacts/experiments/logs/tasks/exp-*.events.jsonl`
- `artifacts/experiments/logs/tasks/exp-*.memory.jsonl`
- `artifacts/experiments/state/runner_state.json`
- `artifacts/experiments/summary/experiment_runs.csv`
- `artifacts/experiments/tasks/*.json`

## 7. Run the Full Experiment Queue

Only start this when the checks above pass and you are ready for a long run.
For unattended execution, prefer the background launcher:

```bash
./.venv/bin/python run_experiments.py launch
```

`launch` writes process output to `artifacts/experiments/logs/background-runner-*.out.log`
and `artifacts/experiments/logs/background-runner-*.err.log` (or the same paths under
`--artifacts-dir`). During long runs, inspect them with:

```bash
tail -f artifacts/experiments/logs/background-runner-*.out.log
tail -f artifacts/experiments/logs/background-runner-*.err.log
```

Foreground mode is still available:

```bash
./.venv/bin/python run_experiments.py run
```

Local dashboard:

```bash
./.venv/bin/streamlit run experiment_dashboard.py
```

The dashboard and CLI share the same persisted runner state.

## 8. Check Status, Stop, and Resume

Check progress:

```bash
./.venv/bin/python run_experiments.py status
```

Request a safe stop:

```bash
./.venv/bin/python run_experiments.py stop
```

Behavior:

- the current experiment is allowed to finish
- completed results stay saved
- the queue stops at the next safe boundary

Resume later:

```bash
./.venv/bin/python run_experiments.py run
```

Useful rerun options:

```bash
./.venv/bin/python run_experiments.py run --rerun-failed
./.venv/bin/python run_experiments.py run --rerun-completed
```

Run exactly one persisted task by id (debugging and future isolated execution):

```bash
./.venv/bin/python run_experiments.py run-task --task-id <TASK_ID>
```

Notes:

- `run-task` rebuilds/syncs the persisted queue before selecting the task id.
- If the task is already completed (including artifact reconciliation when `run_skip` is enabled), it is not re-executed.
- Use the same path/runtime overrides as `run` (`--config`, `--artifacts-dir`, `--raw-data-dir`, and training runtime flags) when reproducing a task.

Reset only the orchestration state:

```bash
./.venv/bin/python run_experiments.py reset
```

Reset state and remove saved run outputs:

```bash
./.venv/bin/python run_experiments.py reset --purge-results
```

## 9. Generate the Final Report

```bash
./.venv/bin/python evaluate.py
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
- Structured run events: `artifacts/experiments/logs/run-events.jsonl`
- Per-task event/memory logs: `artifacts/experiments/logs/tasks/`
- Per-task snapshots: `artifacts/experiments/tasks/*.json`
