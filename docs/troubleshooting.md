# Troubleshooting

This guide is symptom-first and links to authoritative setup/execution references. Commands assume `.venv` is activated.

## Purpose

Provide compact symptom -> cause -> action entries without duplicating full setup or execution manuals.

## Read this when

- A setup, runtime, launch, or report step fails.
- You need the shortest path from observed symptom to corrective action.

## Source of truth

- `scripts/check_environment.py`
- `scripts/check_gpu.py`
- `scripts/check_runtime.py`
- `scripts/check_windows_gpu.py`
- `run_experiments.py`
- `pipeline/experiments/runner.py`
- `evaluate.py`

## Environment check fails

Symptom:

- `python scripts/check_environment.py --require-venv` reports missing packages, invalid Python range, or write-permission issues.

Cause:

- Incomplete bootstrap or unsupported interpreter.

Action:

```bash
python scripts/bootstrap_env.py --gpu auto --dev
python scripts/check_environment.py --require-venv
```

## Dataset validation fails

Symptom:

- `python scripts/validate_dataset.py` reports missing `INbreast.csv` or missing `AllDICOMs/*.dcm`.

Cause:

- Dataset not placed at configured `raw_data_dir`.

Action:

- Place dataset under `data/INbreast Release 1.0/` or update config/CLI path.
- Re-run `python scripts/validate_dataset.py`.

## GPU required checks fail

Symptom:

- `--require-gpu` checks fail, or `probe-runtime --device gpu` fails.

Cause:

- Driver/runtime mismatch, unsupported native Windows TensorFlow version, or no visible GPU.

Action:

- Run:

```bash
python scripts/check_gpu.py --require-gpu
python scripts/check_runtime.py --device gpu --require-gpu
python run_experiments.py probe-runtime --device gpu
```

- For native Windows, follow [windows_gpu_setup.md](windows_gpu_setup.md).
- For WSL/Linux, follow [wsl_gpu_setup.md](wsl_gpu_setup.md).

## Launch behaves unexpectedly with large grids

Symptom:

- Launch startup takes long, rerun flags fail, or queue size is larger than expected.

Cause:

- Large grid dimensions and/or streamed huge-queue mode.

Action:

```bash
python run_experiments.py count
python run_experiments.py launch --max-queue-tasks 50000
python run_experiments.py launch --queue-export-path artifacts/experiments/state/queue-export.jsonl
```

- Narrow grid with `--models`, `--preprocessing`, `--no-combined-preprocessing`, and `--augmentations-per-image`.

## `status` cannot resolve the active runner

Symptom:

- Control command asks you to choose between multiple active runners.

Cause:

- Multiple artifacts roots have active runner PID/state files.

Action:

```bash
python run_experiments.py status --config configs/experiment.low-memory.json
python run_experiments.py status --artifacts-dir artifacts-low-memory
```

## Repeated OOM or unstable long runs

Symptom:

- Tasks fail after several combinations or frequent GPU fallback/OOM events.

Cause:

- Device memory pressure, queue size, or runtime policy mismatches.

Action:

```bash
python scripts/validate_long_runner.py --combinations 20 --device cpu
python run_experiments.py launch --isolate-tasks --device-policy adaptive --gpu-retries 1 --cpu-retries 1
python run_experiments.py status --json
python run_experiments.py partial --max-rows 200
```

- Inspect `artifacts/experiments/logs/run-events.jsonl` and task memory/event logs.

## `evaluate.py` finds no history files

Symptom:

- `evaluate.py` fails with no history CSV files found.

Cause:

- No completed training artifacts in selected history directory/config.

Action:

- Confirm completed runs with `python run_experiments.py status`.
- Confirm history files under `artifacts/runs/history/`.
- Re-run with matching config/artifacts root:

```bash
python evaluate.py --config configs/experiment.smoke.json
```

## Related docs

- Setup and readiness: [setup.md](setup.md)
- Runtime/GPU policy and setup: [gpu.md](gpu.md)
- Runner command reference: [execution.md](execution.md)
- Validation and tests: [testing.md](testing.md)
- Artifact file map: [artifacts.md](artifacts.md)
