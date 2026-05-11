# WSL/Linux GPU Setup

This procedure covers GPU runtime setup for WSL2/Linux environments. It is separate from the native Windows TensorFlow 2.10 path.

## Purpose

Configure and validate TensorFlow GPU support in WSL/Linux using project-supported scripts.

## Read this when

- You run experiments from WSL2 or Linux.
- You need GPU-ready runtime validation before large launches.

## Source of truth

- `scripts/bootstrap_env.py`
- `scripts/check_gpu.py`
- `scripts/check_runtime.py`
- `scripts/install_wsl_cuda.py`
- `pipeline/utils/gpu_env.py`

## Preconditions

1. GPU is visible from Windows host (`nvidia-smi`).
2. GPU is visible from WSL (`/usr/lib/wsl/lib/nvidia-smi`).
3. `.venv` is activated for project commands.

## Commands

Bootstrap with GPU optional:

```bash
python scripts/bootstrap_env.py --gpu auto
```

Bootstrap and require GPU pass:

```bash
python scripts/bootstrap_env.py --gpu required
```

Validate runtime:

```bash
python scripts/check_gpu.py --require-gpu
python scripts/check_runtime.py --device gpu --require-gpu
python run_experiments.py probe-runtime --device gpu
```

## Optional CUDA helper

If you need toolkit installation in WSL, use:

```bash
python scripts/install_wsl_cuda.py
```

Verification-only mode:

```bash
python scripts/install_wsl_cuda.py --verify-only
```

## Outputs

Primary pass/fail signals are `check_gpu.py` and `check_runtime.py` exit codes plus runtime probe output.

## Related docs

- Runtime/device policy overview: [gpu.md](gpu.md)
- Environment bootstrap and checks: [setup.md](setup.md)
- Runner execution controls: [execution.md](execution.md)
- GPU troubleshooting: [troubleshooting.md](troubleshooting.md)
