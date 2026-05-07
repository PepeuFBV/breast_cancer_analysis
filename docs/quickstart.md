# Quick Start

Use the same `--config` for `preprocess.py`, `run_experiments.py launch`,
`status`, `stop`, and `reset`. If you launch `configs/experiment.low-memory.json`,
do not preprocess or check status against the default `artifacts/` tree by
accident.

Windows PowerShell equivalents:

- Python: `.\.venv\Scripts\python.exe`
- Follow logs: `Get-Content <log-path> -Wait`
- Delete artifacts: `Remove-Item <path> -Recurse -Force`

## 0. Native Windows GPU Baseline (TensorFlow 2.10)

Use this only for native Windows CUDA GPU:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows_gpu.ps1 -VenvDir .venv-win-gpu
.\.venv-win-gpu\Scripts\python.exe .\scripts\check_windows_gpu.py
.\.venv-win-gpu\Scripts\python.exe .\run_experiments.py probe-runtime --device gpu
```

If `check_windows_gpu.py` fails, do not launch full experiments.

## 1. Check Runtime

```bash
./.venv/bin/python run_experiments.py probe-runtime --device gpu
./.venv/bin/python run_experiments.py probe-runtime --device cpu
./.venv/bin/python run_experiments.py probe-runtime --device gpu --json
./.venv/bin/python run_experiments.py probe-runtime --device cpu --json
```

Equivalent command:

```bash
python run_experiments.py probe-runtime --device gpu
```

## 1.1 Count Experiments (Dry Run)

```bash
./.venv/bin/python run_experiments.py count
./.venv/bin/python run_experiments.py count --augmentations-per-image 1 2 3
./.venv/bin/python run_experiments.py count --combined-preprocessing
./.venv/bin/python run_experiments.py count --combined-preprocessing --augmentations-per-image 1 2 3
```

Expected totals with default config:

- default: `4,330` experiments / `17,320` fits
- `--augmentations-per-image 1 2 3`: `12,990` experiments / `51,960` fits
- `--combined-preprocessing`: `1,559,530` experiments / `6,238,120` fits
- both flags: `4,678,590` experiments / `18,714,360` fits

## 2. Preprocess

Default artifacts:

```bash
./.venv/bin/python preprocess.py
```

Low-memory config:

```bash
./.venv/bin/python preprocess.py --config configs/experiment.low-memory.json
```

## 3. Start A Safe Adaptive Run

```bash
./.venv/bin/python run_experiments.py launch \
  --config configs/experiment.low-memory.json \
  --isolate-tasks \
  --device-policy adaptive \
  --gpu-retries 1 \
  --cpu-retries 1 \
  --cpu-max-threads 2 \
  --cpu-opencv-threads 1 \
  --cpu-inter-op-threads 1 \
  --cpu-intra-op-threads 2 \
  --cpu-nice 10
```

## 4. Status And Logs

Explicit config:

```bash
./.venv/bin/python run_experiments.py status --config configs/experiment.low-memory.json
```

Plain status also works when there is only one active runner:

```bash
./.venv/bin/python run_experiments.py status
```

Logs:

```bash
tail -f artifacts/experiments/logs/iterative-runner.log
tail -f artifacts-low-memory/experiments/logs/iterative-runner.log
tail -f artifacts-low-memory/experiments/logs/background-runner-*.out.log
tail -f artifacts-low-memory/experiments/logs/background-runner-*.err.log
```

## 5. Stop

Graceful stop:

```bash
./.venv/bin/python run_experiments.py stop --config configs/experiment.low-memory.json
```

Immediate kill:

```bash
./.venv/bin/python run_experiments.py stop --config configs/experiment.low-memory.json --kill
```

## 6. Reset

Reset orchestration state only:

```bash
./.venv/bin/python run_experiments.py reset --config configs/experiment.low-memory.json
```

Reset and delete results:

```bash
./.venv/bin/python run_experiments.py reset --purge-results
./.venv/bin/python run_experiments.py reset --config configs/experiment.low-memory.json --purge-results
```

If a background runner is still alive, kill it as part of reset:

```bash
./.venv/bin/python run_experiments.py reset --config configs/experiment.low-memory.json --purge-results --kill-active
```

## 7. Restart After A New Preprocess Run

```bash
./.venv/bin/python run_experiments.py reset \
  --config configs/experiment.low-memory.json \
  --purge-results \
  --kill-active

./.venv/bin/python preprocess.py --config configs/experiment.low-memory.json

./.venv/bin/python run_experiments.py launch \
  --config configs/experiment.low-memory.json \
  --isolate-tasks \
  --device-policy adaptive \
  --gpu-retries 1 \
  --cpu-retries 1 \
  --cpu-max-threads 2 \
  --cpu-opencv-threads 1 \
  --cpu-inter-op-threads 1 \
  --cpu-intra-op-threads 2 \
  --cpu-nice 10
```

## 8. Smoke Commands

CPU-safe smoke:

```bash
./.venv/bin/python run_experiments.py launch \
  --config configs/experiment.smoke.json \
  --limit 2 \
  --isolate-tasks \
  --device-policy cpu-only \
  --cpu-max-threads 2 \
  --cpu-opencv-threads 1
```

GPU-only validation run:

```bash
python run_experiments.py launch --device-policy gpu-only --limit 1
```

Adaptive smoke:

```bash
./.venv/bin/python run_experiments.py launch \
  --config configs/experiment.smoke.json \
  --limit 3 \
  --isolate-tasks \
  --device-policy adaptive \
  --gpu-retries 1 \
  --cpu-retries 1 \
  --cpu-max-threads 2 \
  --cpu-opencv-threads 1
```

Equivalent adaptive command:

```bash
python run_experiments.py launch --device-policy adaptive --isolate-tasks --limit 10
```

Augmentation dimension command:

```bash
python run_experiments.py launch --augmentations-per-image 1 2 3 --limit 10
```

GPU probe command:

```bash
python run_experiments.py probe-runtime --device gpu
```

## 9. Huge Queues

The runner refuses very large queues by default.

- Narrow with `--models`, `--preprocessing`, and `--no-combined-preprocessing`
- Export the queue first with `--queue-export-path`
- Use `--allow-huge-queue` only intentionally

Example:

```bash
./.venv/bin/python run_experiments.py launch \
  --allow-huge-queue \
  --queue-export-path artifacts/experiments/state/queue-export.jsonl
```
