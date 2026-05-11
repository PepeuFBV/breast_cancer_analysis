# Artifacts

This document defines the output and state directory layout under the configured artifacts root. Use it to inspect run progress and diagnose failures.

## Purpose

Map pipeline stages to artifact files and identify the highest-value files for run inspection.

## Read this when

- You need to locate outputs from preprocessing, training, runner orchestration, or evaluation.
- You need to inspect state/log files during active runs.

## Source of truth

- `pipeline/utils/paths.py`
- `preprocess.py`
- `train.py`
- `run_experiments.py`
- `pipeline/experiments/runner.py`
- `evaluate.py`
- `pipeline/evaluate/reporting.py`

## Artifacts root

Default root is `artifacts/`, configurable via:

- config: `paths.artifacts_dir`
- CLI: `--artifacts-dir` where supported

## Outputs

### Processed artifacts

- `artifacts/processed/images/`
- `artifacts/processed/splits/train_split.csv`
- `artifacts/processed/splits/test_split.csv`
- `artifacts/processed/splits/split_summary.json`

### Training run artifacts

- `artifacts/runs/history/<preproc_id>/<model_name>/history_<param>__aug<N>.csv`
- `artifacts/runs/predictions/<preproc_id>/<model_name>/<param>__aug<N>.csv`

### Experiment orchestration state

- `artifacts/experiments/state/runner_state.json`
- `artifacts/experiments/summary/experiment_runs.csv`
- `artifacts/experiments/tasks/*.json`
- `artifacts/experiments/control/runner_pid.json`
- `artifacts/experiments/control/stop_requested.flag`
- `artifacts/experiments/control/runner_lifecycle.json`

### Logs

- `artifacts/experiments/logs/iterative-runner.log`
- `artifacts/experiments/logs/run-events.jsonl`
- `artifacts/experiments/logs/background-runner-*.out.log`
- `artifacts/experiments/logs/background-runner-*.err.log`
- `artifacts/experiments/logs/tasks/*.log`
- `artifacts/experiments/logs/tasks/*.events.jsonl`
- `artifacts/experiments/logs/tasks/*.memory.jsonl`

### Reports

- `artifacts/reports/final_comprehensive_results.csv`
- `artifacts/reports/evaluation_details/...` (or configured details dir)

## Inspect a run quickly

1. Summary counters and current task:

```bash
python run_experiments.py status --json
```

2. Completed rows so far:

```bash
python run_experiments.py partial --max-rows 200
```

3. Timeline and failure events:

- `artifacts/experiments/logs/run-events.jsonl`
- `artifacts/experiments/logs/tasks/*.events.jsonl`
- `artifacts/experiments/logs/tasks/*.memory.jsonl`

## Related docs

- Runner command behavior: [execution.md](execution.md)
- Configured paths and overrides: [configuration.md](configuration.md)
- Evaluation/testing workflow: [testing.md](testing.md)
- Failure diagnosis entrypoint: [troubleshooting.md](troubleshooting.md)
