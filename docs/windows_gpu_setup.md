# Native Windows 11 GPU Setup (TensorFlow 2.10)

This project can run GPU on native Windows only with the legacy TensorFlow CUDA
path:

- Python `3.10.x`
- TensorFlow `2.10.x`
- CUDA Toolkit `11.2`
- cuDNN `8.1` (`cudnn64_8.dll`)

## 1. Create a Python 3.10 Virtualenv

```powershell
py -3.10 -m venv .venv-win-gpu
.\.venv-win-gpu\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

## 2. Install Project Dependencies

```powershell
.\.venv-win-gpu\Scripts\python.exe -m pip install -r requirements-windows-gpu.txt
.\.venv-win-gpu\Scripts\python.exe -m pip install -e .
```

Or use the helper:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows_gpu.ps1 -VenvDir .venv-win-gpu
```

## 3. Install CUDA and cuDNN

1. Install NVIDIA driver (current stable).
2. Install CUDA Toolkit `11.2`.
3. Install cuDNN `8.1` for CUDA `11.2`.
4. Confirm `%PATH%` includes CUDA directories, for example:
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp`

The runtime now auto-prepends the TensorFlow 2.10-compatible CUDA 11.2
directories from the standard install location before importing TensorFlow.
This avoids the common slow fallback path where the machine-wide `%PATH%`
points at a newer CUDA release such as `v11.8`, which TensorFlow 2.10 cannot
use on native Windows.

## 4. Validate Runtime

```powershell
.\.venv-win-gpu\Scripts\python.exe .\scripts\check_windows_gpu.py
.\.venv-win-gpu\Scripts\python.exe .\run_experiments.py probe-runtime --device gpu
```

Do not proceed to full runs unless both checks pass and TensorFlow reports at
least one GPU device.

## 5. Smoke Validation Before Large Queue

```powershell
.\.venv-win-gpu\Scripts\python.exe .\run_experiments.py launch --device-policy gpu-only --isolate-tasks --limit 1
.\.venv-win-gpu\Scripts\python.exe .\run_experiments.py launch --device-policy adaptive --isolate-tasks --include-combinations --limit 2
.\.venv-win-gpu\Scripts\python.exe .\run_experiments.py dry-run --include-combinations
```

Expected dry-run totals:

- `1,559,530` experiments
- `6,238,120` fits

## 6. Launch Full Combinations Queue (Only After GPU Passes)

```powershell
.\.venv-win-gpu\Scripts\python.exe .\run_experiments.py stop
.\.venv-win-gpu\Scripts\python.exe .\run_experiments.py status
.\.venv-win-gpu\Scripts\python.exe .\run_experiments.py launch --include-combinations --device-policy adaptive --isolate-tasks --allow-huge-queue
```

Do not add `--limit` to the launch command.
