# WSL GPU Setup for TensorFlow

This guide covers the additional steps needed to enable TensorFlow GPU support in WSL2.

## Prerequisites

- Windows 11 or Windows 10 version 21H2 or higher
- WSL2 installed and configured
- NVIDIA GPU with recent drivers installed on Windows host
- Ubuntu or Debian-based WSL2 distribution

## Verify Windows GPU Driver

From PowerShell or Command Prompt on Windows:

```powershell
nvidia-smi
```

This should show your GPU. The Windows driver version should be 510.06 or newer for CUDA 11.6+ support.

## Verify WSL Can See the GPU

From your WSL2 terminal:

```bash
/usr/lib/wsl/lib/nvidia-smi
```

If this works, WSL GPU passthrough is functional. If not, update your Windows GPU drivers and ensure WSL2 is up to date:

```powershell
wsl --update
```

## Install CUDA Toolkit in WSL (Optional but Recommended)

The `tensorflow[and-cuda]` package includes CUDA libraries, but installing the CUDA toolkit can help with compatibility.

### Automated Installation

Use the provided helper script:

```bash
python3 scripts/install_wsl_cuda.py
```

This will:
1. Verify GPU visibility via nvidia-smi
2. Install CUDA toolkit 12.3 (compatible with TensorFlow 2.16+)
3. Install cuDNN libraries
4. Verify the installation

For a different CUDA version:

```bash
python3 scripts/install_wsl_cuda.py --cuda-version 11-8
```

To skip cuDNN (tensorflow[and-cuda] includes it):

```bash
python3 scripts/install_wsl_cuda.py --skip-cudnn
```

### Manual Installation

Alternatively, install manually:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-3
```

Note: TensorFlow 2.16+ supports CUDA 12.3. Adjust the version if using a different TensorFlow version.

## Bootstrap with GPU Support

After verifying GPU visibility:

```bash
python3 scripts/bootstrap_env.py --gpu auto
```

Or require GPU to fail fast if it's not working:

```bash
python3 scripts/bootstrap_env.py --gpu required
```

## Verify TensorFlow GPU

```bash
./.venv/bin/python scripts/check_gpu.py --require-gpu
```

If this fails with "No TensorFlow GPU devices are visible", the most common causes are:

1. **CUDA/cuDNN version mismatch**: The `tensorflow[and-cuda]` package should handle this automatically, but older TensorFlow versions may need manual CUDA installation.

2. **Missing cuDNN libraries**: Install cuDNN if needed:
   ```bash
   sudo apt install libcudnn8 libcudnn8-dev
   ```

3. **WSL GPU passthrough not enabled**: Update Windows and WSL2 to the latest versions.

## Running Without GPU

If GPU setup is problematic, the pipeline runs fine on CPU:

```bash
python3 scripts/bootstrap_env.py --gpu off
```

All experiments will use CPU, which is slower but fully functional.

## Troubleshooting

### TensorFlow sees CUDA build but no GPU devices

This means TensorFlow was built with CUDA support but cannot load the runtime libraries. Solutions:

1. Reinstall with the CUDA-enabled package:
   ```bash
   python3 scripts/bootstrap_env.py --gpu auto
   ```

2. Verify the NVIDIA libraries are accessible:
   ```bash
   ls -la /usr/lib/wsl/lib/libcuda.so*
   ls -la /usr/lib/wsl/lib/libnvidia*.so*
   ```

3. Check that the environment bootstrap applied the library paths:
   ```bash
   ./.venv/bin/python -c "import os; print(os.environ.get('LD_LIBRARY_PATH', 'not set'))"
   ```

### GPU works in nvidia-smi but not in TensorFlow

The Windows NVIDIA driver provides GPU access, but TensorFlow needs compatible CUDA/cuDNN libraries in the WSL environment. Install the CUDA toolkit as shown above.

### Performance is slower than expected

1. Verify GPU is actually being used:
   ```bash
   watch -n 1 /usr/lib/wsl/lib/nvidia-smi
   ```
   Run training in another terminal and watch for GPU utilization.

2. Check for memory constraints:
   ```bash
   ./.venv/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

3. Ensure batch size is appropriate for your GPU memory in `configs/experiment.default.json`.
