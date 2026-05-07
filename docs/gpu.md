# GPU Validation

TensorFlow can run on CPU. GPU is optional unless you explicitly require it for
an experiment.

## Native Windows 11 (TensorFlow 2.10)

For native Windows GPU, use the TensorFlow 2.10 stack only:

- Python `3.10.x`
- TensorFlow `2.10.x`
- CUDA `11.2`
- cuDNN `8.1`

Quick setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows_gpu.ps1 -VenvDir .venv-win-gpu
.\.venv-win-gpu\Scripts\python.exe .\scripts\check_windows_gpu.py
.\.venv-win-gpu\Scripts\python.exe .\run_experiments.py probe-runtime --device gpu
```

If TensorFlow is `2.11+`, native Windows CUDA GPU is unsupported and the probe
must fail until the stack is downgraded.

Recommended setup on Linux/WSL2:

```bash
python3 scripts/bootstrap_env.py --gpu auto
```

Require a working GPU stack during bootstrap:

```bash
python3 scripts/bootstrap_env.py --gpu required
```

The bootstrap script creates or repairs `.venv`, installs the project
dependencies, and switches the TensorFlow requirement to
`tensorflow[and-cuda]` when an NVIDIA driver is visible from Linux/WSL.

For detailed WSL2 GPU setup instructions, see [`wsl_gpu_setup.md`](wsl_gpu_setup.md).
For native Windows setup instructions, see [`windows_gpu_setup.md`](windows_gpu_setup.md).

CPU/GPU optional check:

```bash
./.venv/bin/python scripts/check_gpu.py
./.venv/bin/python scripts/check_runtime.py --device auto
```

CPU-only check:

```bash
./.venv/bin/python scripts/check_gpu.py --cpu-only
./.venv/bin/python scripts/check_runtime.py --device cpu
```

Require a visible GPU:

```bash
./.venv/bin/python scripts/check_gpu.py --require-gpu
./.venv/bin/python scripts/check_runtime.py --device gpu --require-gpu
```

If you want GPU when available but do not want the check to fail on CPU-only
machines, use:

```bash
./.venv/bin/python scripts/check_runtime.py --device gpu
```

Device policy behavior in `run_experiments.py run`:

- `gpu-only`: requires a visible and usable GPU before launching tasks; fails fast when GPU is unavailable.
- `gpu-first`: starts on GPU, then falls back to CPU for task-level GPU failure when retries are exhausted.
- `adaptive`: GPU-first with CPU fallback and GPU recovery probes after cooldown.
- `cpu-only`: forces `CUDA_VISIBLE_DEVICES=-1` and never schedules GPU attempts.

Runtime probe command:

```bash
python run_experiments.py probe-runtime --device gpu
```

The required check fails when TensorFlow cannot see a GPU. That usually means
one of these is missing or mismatched:

- NVIDIA driver
- CUDA/cuDNN libraries compatible with the installed TensorFlow wheel
- WSL2 GPU passthrough, when using WSL
- `CUDA_VISIBLE_DEVICES` configuration

The project includes WSL CUDA library path bootstrapping in
`pipeline/utils/gpu_env.py`, but the host driver and TensorFlow-compatible CUDA
stack still need to be installed correctly.
