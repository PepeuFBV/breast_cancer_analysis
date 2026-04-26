# Troubleshooting

## `No module named venv`

Install the venv package:

```bash
sudo apt install python3-venv
```

## `pip` Is Missing

Install pip:

```bash
sudo apt install python3-pip
```

## Build or Wheel Errors

Install build tools and Python headers:

```bash
sudo apt install build-essential python3-dev
```

Also confirm you are using Python 3.10, 3.11, or 3.12:

```bash
python --version
```

## Dataset Validation Fails

Run:

```bash
python scripts/validate_dataset.py
```

The expected layout is:

```text
data/INbreast Release 1.0/
  INbreast.csv
  AllDICOMs/*.dcm
```

## TensorFlow Imports but GPU Is Not Visible

Run:

```bash
python scripts/check_gpu.py
python scripts/check_gpu.py --require-gpu
```

Optional mode exits successfully on CPU and prints a warning. Required mode
fails if no GPU is visible.

## Runner Stops or Some Tasks Fail

Check status and logs:

```bash
python run_experiments.py status
python run_experiments.py status --json
```

Key files:

- `artifacts/experiments/state/runner_state.json`
- `artifacts/experiments/summary/experiment_runs.csv`
- `artifacts/experiments/logs/iterative-runner.log`

Failed tasks are not rerun by default:

```bash
python run_experiments.py run --rerun-failed
```

Reset only orchestration state:

```bash
python run_experiments.py reset
```

Reset state and saved run outputs:

```bash
python run_experiments.py reset --purge-results
```
