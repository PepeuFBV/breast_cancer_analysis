# Configuration

This document explains the experiment configuration model used by `configs/*.json`. It focuses on the fields needed to run and control experiments.

## Purpose

Define how config sections map to runtime behavior and how CLI flags override config values.

## Read this when

- You are editing `configs/experiment.default.json`, `configs/experiment.low-memory.json`, or `configs/experiment.smoke.json`.
- You need to understand CLI overrides versus config defaults.

## Source of truth

- `pipeline/config.py`
- `configs/experiment.default.json`
- `configs/experiment.low-memory.json`
- `configs/experiment.smoke.json`
- `train.py`
- `preprocess.py`
- `evaluate.py`
- `run_experiments.py`

## Config files

- `configs/experiment.default.json`: full default grid.
- `configs/experiment.low-memory.json`: constrained grid + CPU limits for lower memory pressure.
- `configs/experiment.smoke.json`: minimal fast validation grid.

## Section overview

### `paths`

Controls raw dataset root and artifacts root, plus optional explicit overrides for split/history/prediction/report paths.

- Relative paths are resolved from repository root.
- If optional path fields are `null`, code uses defaults from `pipeline/utils/paths.py`.

### `preprocess`

Controls dataset preparation defaults.

- `image_size`
- `augmentations_per_image`
- `samples_per_class`
- `test_size`
- `random_state`
- `preprocessing_grid`

`preprocessing_grid` defines parameter search spaces per preprocessing id (`none`, `denoise`, `binarize`, `lowpass`, `erode`, `dilate`, `open`, `close`, `clahe`).

### `train`

Controls training/task dimensions and baseline training hyperparameters.

- `model_names`
- `preprocessing_ids`
- `include_combinations`
- `folds`
- `validation_size`
- `batch_size`
- `epochs`
- `learning_rate`
- `loss`
- `run_skip`
- `random_state`

### `runner`

Controls iterative orchestration behavior for `run_experiments.py launch`.

Includes:

- lifecycle and isolation (`isolate_tasks`, `task_cooldown_seconds`, `task_timeout_seconds`)
- device strategy (`device_policy`, `gpu_retries`, `cpu_retries`)
- OOM behavior (`cooldown_after_oom_seconds`, `gpu_recovery_cooldown_seconds`, `max_consecutive_oom`, `max_task_attempts`, `fail_fast_on_oom`)
- optional thermal policy fields
- optional CPU limit fields (`cpu_max_threads`, `cpu_opencv_threads`, `cpu_inter_op_threads`, `cpu_intra_op_threads`, `cpu_nice`)

### `models`

Per-model runtime overrides consumed by model builders.

Typical keys:

- `input_channels`
- `batch_size`
- `dense_units`
- `dropout_rate`
- `dropout_rates`

### `evaluate`

Controls evaluation report behavior.

- `top_k`

## CLI override precedence

General precedence is:

1. Explicit CLI flags
2. Config file values
3. Built-in defaults in code

Examples:

- `preprocess.py` CLI overrides selected `preprocess` and `paths` values.
- `train.py` and `run_experiments.py` training flags override `train` and selected `paths` values.
- `evaluate.py` flags override evaluate output/history/prediction/detail paths and `top_k`.
- `run_experiments.py launch` flags override `runner` values.

Boolean note:

- `--fail-fast-on-oom` and `--thermal-policy-enabled` are enable-only flags at CLI level; they cannot force-disable a `true` config value.

## Related docs

- Runner command reference: [execution.md](execution.md)
- Setup and dataset checks: [setup.md](setup.md)
- Fast run paths: [quickstart.md](quickstart.md)
- Output layout and inspection files: [artifacts.md](artifacts.md)
