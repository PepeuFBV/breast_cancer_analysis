# Quickstart

This document gives the shortest safe paths from setup to a successful run. Commands assume `.venv` is activated.

## Purpose

Get from clone/setup to a verified first run with minimal commands, then point to detailed docs for control and tuning.

## Read this when

- You want a first successful execution path.
- You want unattended setup as a fast bootstrap option.

## Source of truth

- `scripts/bootstrap_env.py`
- `scripts/unattended_setup.py`
- `scripts/check_readiness.py`
- `preprocess.py`
- `run_experiments.py`
- `evaluate.py`
- `configs/experiment.smoke.json`

## Path 1: Manual first run

1. Bootstrap environment.

```bash
python scripts/bootstrap_env.py --gpu auto
```

2. Validate dataset layout.

```bash
python scripts/validate_dataset.py
```

3. Build processed splits/images.

```bash
python preprocess.py
```

4. Inspect queue size before launching.

```bash
python run_experiments.py count --config configs/experiment.smoke.json
```

5. Launch a small smoke run.

```bash
python run_experiments.py launch --config configs/experiment.smoke.json --isolate-tasks --device-policy adaptive
```

6. Inspect current status and partial results.

```bash
python run_experiments.py status --config configs/experiment.smoke.json
python run_experiments.py partial --config configs/experiment.smoke.json --max-rows 50
```

7. Generate evaluation report for that config/artifacts tree.

```bash
python evaluate.py --config configs/experiment.smoke.json
```

## Path 2: Unattended bootstrap path

1. Run unattended setup (bootstrap + dataset validation + smoke + preprocess).

```bash
python scripts/unattended_setup.py --gpu auto
```

2. Check readiness summary.

```bash
python scripts/check_readiness.py
```

3. Start and monitor execution.

```bash
python run_experiments.py launch
python run_experiments.py status
```

## Related docs

- Setup details and prerequisites: [setup.md](setup.md)
- Full runner command reference: [execution.md](execution.md)
- Config structure and overrides: [configuration.md](configuration.md)
- Runtime and GPU policy: [gpu.md](gpu.md)
