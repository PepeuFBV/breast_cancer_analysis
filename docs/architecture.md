# Architecture

This is a concise mental model of repository structure and runtime flow. It focuses on entrypoints, pipeline modules, configs, scripts, and artifacts.

## Purpose

Help maintainers and automation quickly locate where behavior is implemented.

## Read this when

- You need orientation before changing pipeline behavior or docs.
- You need to map commands to implementation modules.

## Source of truth

- `preprocess.py`
- `train.py`
- `run_experiments.py`
- `evaluate.py`
- `pipeline/config.py`
- `pipeline/data/`
- `pipeline/train/`
- `pipeline/experiments/`
- `pipeline/evaluate/`
- `pipeline/utils/paths.py`
- `configs/*.json`

## Entry points

- `preprocess.py`: build processed images/splits from raw INbreast layout.
- `train.py`: run direct training tasks from processed splits.
- `run_experiments.py`: iterative runner lifecycle (`probe-runtime`, `count`, `launch`, `status`, `partial`, `stop`, `reset`).
- `evaluate.py`: aggregate run artifacts into final reports.

## Pipeline modules

- `pipeline/data`: dataset validation, split logic, image export/augmentation.
- `pipeline/train`: preprocessing task expansion, model builders, training execution, artifact writing.
- `pipeline/experiments`: iterative queue/state/log orchestration and device-policy execution.
- `pipeline/evaluate`: report aggregation and per-run evaluation details.
- `pipeline/utils`: runtime/device helpers, GPU env bootstrap, memory and path utilities.

## Configs and scripts

- `configs/*.json`: runtime defaults per profile (`default`, `low-memory`, `smoke`).
- `scripts/bootstrap_env.py`: environment bootstrap and optional GPU validation.
- `scripts/check_*.py`: environment/runtime/GPU/readiness checks.
- `scripts/smoke_run.py`: synthetic smoke pipeline.
- `scripts/validate_long_runner.py`: tiny long-run orchestration validation.

## Artifacts

Default output root is `artifacts/` (overridable by config or CLI), containing:

- processed dataset artifacts
- training run history/predictions
- experiment state/summary/logs/control files
- final reports and evaluation details

## Related docs

- Docs map: [index.md](index.md)
- Config model: [configuration.md](configuration.md)
- Runner lifecycle: [execution.md](execution.md)
- Output layout: [artifacts.md](artifacts.md)
