# GPU and Runtime

This document explains runtime device behavior and GPU requirement levels for this project. It is the policy overview, not the OS-specific install procedure.

## Purpose

Clarify optional versus required GPU usage, runtime probe commands, and device policy behavior in `run_experiments.py`.

## Read this when

- You need to decide between CPU, GPU-first, adaptive fallback, or GPU-only execution.
- You need to validate runtime visibility before launching experiments.

## Source of truth

- `scripts/bootstrap_env.py`
- `scripts/check_gpu.py`
- `scripts/check_runtime.py`
- `run_experiments.py`
- `pipeline/utils/runtime_device.py`
- `pipeline/utils/runtime_probe.py`

## Optional GPU vs required GPU

- GPU is optional for pipeline execution unless you explicitly require it.
- `bootstrap_env.py --gpu auto` installs a working runtime for CPU/GPU-available systems.
- `bootstrap_env.py --gpu required` fails setup when GPU validation cannot pass.

## Runtime probe commands

```bash
python scripts/check_gpu.py
python scripts/check_runtime.py --device auto
python run_experiments.py probe-runtime --device auto
```

Require visible and usable GPU:

```bash
python scripts/check_gpu.py --require-gpu
python scripts/check_runtime.py --device gpu --require-gpu
python run_experiments.py probe-runtime --device gpu
```

CPU-only validation:

```bash
python scripts/check_gpu.py --cpu-only
python scripts/check_runtime.py --device cpu
python run_experiments.py probe-runtime --device cpu
```

## Device policies in `launch`

- `cpu-only`: run tasks on CPU only.
- `gpu-only`: require usable GPU; fail if GPU policy cannot be satisfied.
- `gpu-first`: prefer GPU, with CPU fallback when retries are exhausted.
- `adaptive`: GPU-first plus recovery probes and CPU fallback behavior.

Example:

```bash
python run_experiments.py launch --device-policy adaptive --isolate-tasks
```

## Windows and WSL setup locations

- Native Windows TensorFlow 2.10 path: [windows_gpu_setup.md](windows_gpu_setup.md)
- WSL/Linux GPU path: [wsl_gpu_setup.md](wsl_gpu_setup.md)

## Related docs

- Environment bootstrap and readiness: [setup.md](setup.md)
- Runner command details: [execution.md](execution.md)
- Quick launch paths: [quickstart.md](quickstart.md)
- GPU-related failures: [troubleshooting.md](troubleshooting.md)
