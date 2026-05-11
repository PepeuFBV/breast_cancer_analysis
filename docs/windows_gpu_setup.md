# Native Windows GPU Setup

This procedure is for native Windows TensorFlow GPU only. It uses the TensorFlow `2.10.x` compatibility path implemented by project scripts.

## Purpose

Provide the shortest supported native Windows GPU setup and validation sequence.

## Read this when

- You are running on native Windows (not WSL).
- You need TensorFlow GPU support with the project's Windows constraints.

## Source of truth

- `scripts/bootstrap_windows_gpu.ps1`
- `scripts/check_windows_gpu.py`
- `pipeline/utils/runtime_device.py`
- `docs/gpu.md`

## Constraints

Use this stack:

- Python `3.10.x`
- TensorFlow `2.10.x`
- CUDA `11.2`
- cuDNN `8.1`

## Commands

Preferred scripted setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows_gpu.ps1 -VenvDir .venv
```

Manual validation:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_windows_gpu.py
.\.venv\Scripts\python.exe .\run_experiments.py probe-runtime --device gpu
```

Optional smoke launch after checks pass:

```powershell
.\.venv\Scripts\python.exe .\run_experiments.py launch --config configs/experiment.smoke.json --isolate-tasks --device-policy gpu-only --limit 1
```

## Outputs

Use `check_windows_gpu.py` output and `probe-runtime` output as the baseline pass/fail criteria before large queues.

## Related docs

- Runtime/device policy overview: [gpu.md](gpu.md)
- Runner command reference: [execution.md](execution.md)
- WSL/Linux setup alternative: [wsl_gpu_setup.md](wsl_gpu_setup.md)
- GPU troubleshooting: [troubleshooting.md](troubleshooting.md)
