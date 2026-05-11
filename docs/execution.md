# Execution

This is the authoritative reference for `run_experiments.py` command behavior. It covers queue sizing, launch lifecycle, status inspection, and run control.

## Purpose

Describe how to run and control iterative experiments using code-backed command semantics.

## Read this when

- You are launching, monitoring, stopping, or resetting experiment runs.
- You need `run_experiments.py` subcommand and flag details.

## Source of truth

- `run_experiments.py`
- `pipeline/experiments/runner.py`
- `train.py`
- `configs/experiment.default.json`
- `configs/experiment.low-memory.json`
- `configs/experiment.smoke.json`

## Command scope and selection

By default, control commands resolve an active runner automatically. If multiple active runners exist, pass `--config` or `--artifacts-dir` explicitly.

Common selectors:

```bash
python run_experiments.py status --config configs/experiment.low-memory.json
python run_experiments.py status --artifacts-dir artifacts-low-memory
```

## Commands

### `probe-runtime`

Check TensorFlow/runtime visibility before launch.

```bash
python run_experiments.py probe-runtime --device auto
python run_experiments.py probe-runtime --device gpu --json
python run_experiments.py probe-runtime --device cpu --cpu-max-threads 2 --cpu-opencv-threads 1
```

### `count` (`dry-run` alias)

Estimate preprocessing/model/augmentation dimensions and total fits.

```bash
python run_experiments.py count
python run_experiments.py count --config configs/experiment.low-memory.json
python run_experiments.py count --models "custom cnn" --preprocessing none --augmentations-per-image 1 2
```

### `launch`

Start background execution.

```bash
python run_experiments.py launch
```

Safe adaptive example (low-memory config):

```bash
python run_experiments.py launch --config configs/experiment.low-memory.json --isolate-tasks --device-policy adaptive --gpu-retries 1 --cpu-retries 1 --cpu-max-threads 2 --cpu-opencv-threads 1 --cpu-inter-op-threads 1 --cpu-intra-op-threads 2 --cpu-nice 10
```

### `status`

Show execution summary.

```bash
python run_experiments.py status
python run_experiments.py status --json
```

### `partial`

Extract completed-results-so-far from persisted summary/state.

```bash
python run_experiments.py partial
python run_experiments.py partial --max-rows 200 --csv-output artifacts/experiments/summary/partial.csv
```

### `stop`

Request graceful stop or terminate active process.

```bash
python run_experiments.py stop
python run_experiments.py stop --kill
```

### `reset`

Reset runner metadata, optionally purge training outputs.

```bash
python run_experiments.py reset
python run_experiments.py reset --purge-results
python run_experiments.py reset --purge-results --kill-active
```

## Smoke runs

Use smoke config for short validation loops.

```bash
python run_experiments.py launch --config configs/experiment.smoke.json --limit 3 --isolate-tasks --device-policy adaptive
python run_experiments.py status --config configs/experiment.smoke.json
```

## Huge queue precautions

- Always run `count` first.
- Limit dimensions with `--models`, `--preprocessing`, `--no-combined-preprocessing`, `--augmentations-per-image`.
- Use `--max-queue-tasks <N>` when you want an explicit hard cap.
- Use `--queue-export-path <path>` to inspect/export queue records.

Examples:

```bash
python run_experiments.py launch --max-queue-tasks 50000
python run_experiments.py launch --queue-export-path artifacts/experiments/state/queue-export.jsonl
```

Behavior notes from current code:

- When `max_queue_tasks` is unset, very large queues may run in streamed mode (instead of full in-memory materialization).
- `--rerun-failed` and `--rerun-completed` are not supported in streamed huge-queue mode.

## Outputs

Key runtime files (under the selected artifacts root):

- `artifacts/experiments/state/runner_state.json`
- `artifacts/experiments/summary/experiment_runs.csv`
- `artifacts/experiments/logs/iterative-runner.log`
- `artifacts/experiments/logs/run-events.jsonl`
- `artifacts/experiments/logs/tasks/*.events.jsonl`

## Related docs

- Fast run paths: [quickstart.md](quickstart.md)
- Config semantics and override precedence: [configuration.md](configuration.md)
- Output file layout: [artifacts.md](artifacts.md)
- Runtime/GPU policy: [gpu.md](gpu.md)
- Failure handling guide: [troubleshooting.md](troubleshooting.md)
