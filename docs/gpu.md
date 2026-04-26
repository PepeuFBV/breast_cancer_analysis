# GPU Validation

TensorFlow can run on CPU. GPU is optional unless you explicitly require it for
an experiment.

CPU/GPU optional check:

```bash
python scripts/check_gpu.py
```

CPU-only check:

```bash
python scripts/check_gpu.py --cpu-only
```

Require a visible GPU:

```bash
python scripts/check_gpu.py --require-gpu
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
