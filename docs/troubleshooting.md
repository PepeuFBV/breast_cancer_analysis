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

Repair the environment first:

```bash
python3 scripts/bootstrap_env.py --gpu auto
```

If the run must use GPU:

```bash
python3 scripts/bootstrap_env.py --gpu required
```

Then re-run:

```bash
./.venv/bin/python scripts/check_gpu.py
./.venv/bin/python scripts/check_gpu.py --require-gpu
```

And verify runner probe output:

```bash
python run_experiments.py probe-runtime --device gpu
```

Optional mode exits successfully on CPU and prints a warning. Required mode
fails if no GPU is visible.

For WSL2-specific GPU setup, see [`wsl_gpu_setup.md`](wsl_gpu_setup.md).

### Native Windows 11 + TensorFlow

Native Windows CUDA GPU requires TensorFlow `2.10.x` only.

If `python run_experiments.py probe-runtime --device gpu` reports an
unsupported stack with TensorFlow `2.11+`, downgrade using:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows_gpu.ps1 -VenvDir .venv
.\.venv\Scripts\python.exe .\scripts\check_windows_gpu.py
.\.venv\Scripts\python.exe .\run_experiments.py probe-runtime --device gpu
```

Checklist for TensorFlow 2.10 native Windows path:

- Python `3.10.x`
- TensorFlow `2.10.x`
- CUDA Toolkit `11.2`
- cuDNN `8.1` (`cudnn64_8.dll`)
- CUDA bin directories present in `%PATH%`

If the checker reports missing CUDA/cuDNN DLLs, fix those first and rerun
`scripts/check_windows_gpu.py` before any `gpu-only` run.

## TensorFlow Is Missing or Fails to Import

Run:

```bash
python3 scripts/bootstrap_env.py --gpu auto
```

That command recreates or repairs `.venv`, installs the project requirements,
and re-runs the setup checks.

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
- `artifacts/experiments/logs/run-events.jsonl`
- `artifacts/experiments/logs/tasks/exp-*.events.jsonl`
- `artifacts/experiments/logs/tasks/exp-*.memory.jsonl`
- `artifacts/experiments/logs/tasks/exp-*.log`
- `artifacts/experiments/tasks/*.json`

If failures appear after several combinations, inspect memory snapshots in
`iterative-runner.log` (`[memory] before:*` and `[memory] after:*`) and compare
task-level JSON snapshots to identify where failures started.

### Structured Memory Diagnostics

Use `run-events.jsonl` for global timeline and `logs/tasks/` for per-task detail.

1. Find recent failures:

```bash
tail -n 200 artifacts/experiments/logs/run-events.jsonl | rg '"phase":"task:failed"'
```

2. Inspect one failed task timeline:

```bash
task_id="exp-<id>"
rg '"phase":"(task:start|after_cleanup|task:failed)"' \
  "artifacts/experiments/logs/tasks/${task_id}.memory.jsonl"
```

3. Compare memory drift between task boundaries:

- `task:start` shows baseline before training.
- `after_cleanup` shows post-release memory for that task.
- `task:failed` captures memory at failure with error metadata and traceback summary.

Failed tasks are not rerun by default:

```bash
python run_experiments.py launch --rerun-failed
```

Reset only orchestration state:

```bash
python run_experiments.py reset
```

Reset state and saved run outputs:

```bash
python run_experiments.py reset --purge-results
```

## Long Runs Fail After 8-10 Combinations

Use smoke validation instead of jumping directly to the full queue:

```bash
./.venv/bin/python scripts/check_runtime.py --device auto
./.venv/bin/python scripts/validate_long_runner.py --combinations 20 --device cpu
```

CPU-only mode is supported:

```bash
./.venv/bin/python scripts/check_runtime.py --device cpu
```

GPU is optional unless `--require-gpu` is passed.

Use the long-run validator before retrying the full queue:

```bash
./.venv/bin/python scripts/validate_long_runner.py --combinations 20 --device auto
```

For production long runs, use adaptive per-task subprocess policy:

```bash
./.venv/bin/python run_experiments.py launch \
  --isolate-tasks \
  --device-policy adaptive \
  --gpu-retries 1 \
  --cpu-retries 1 \
  --cooldown-after-oom-seconds 15 \
  --gpu-recovery-cooldown-seconds 60 \
  --max-consecutive-oom 3
```

GPU-only policy must fail when GPU is unavailable:

```bash
python run_experiments.py launch --device-policy gpu-only --limit 1
```

Quick adaptive smoke:

```bash
python run_experiments.py launch --device-policy adaptive --isolate-tasks --limit 10
```

Augmentation dimension smoke:

```bash
python run_experiments.py launch --augmentations-per-image 1 2 3 --limit 10
```

Count large grids before running:

```bash
python run_experiments.py count --combined-preprocessing --augmentations-per-image 1 2 3
```

How adaptive fallback behaves:

- GPU is preferred for each new task.
- GPU OOM retries happen in fresh subprocesses.
- If GPU keeps failing, the same task falls back to CPU (`CUDA_VISIBLE_DEVICES=-1`).
- Later tasks retry GPU after recovery cooldown.
- Runner stops safely when `--max-consecutive-oom` is reached.

Inspect OOM/fallback attempts:

```bash
rg '"event":"(gpu_oom_detected|gpu_retry_scheduled|cpu_fallback_scheduled|cpu_fallback_succeeded|gpu_recovery_probe_scheduled|gpu_recovered|oom_policy_stop|task_attempt_finished)"' \
  artifacts/experiments/logs/run-events.jsonl
```
