from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WINDOWS_TF_VERSION_PREFIX = "2.10."
REQUIRED_PYTHON = (3, 10)
REQUIRED_CUDA_VERSION_HINT = "11.2"
REQUIRED_CUDNN_DLL = "cudnn64_8.dll"
REQUIRED_CUDA_DLLS = (
    "cudart64_110.dll",
    "cublas64_11.dll",
    "cublasLt64_11.dll",
    "cufft64_10.dll",
    "curand64_10.dll",
    "cusolver64_11.dll",
    "cusparse64_11.dll",
)


@dataclass(frozen=True)
class WindowsGpuCheckResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    details: dict[str, Any]


def _parse_major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _run_command(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return int(completed.returncode), completed.stdout.strip(), completed.stderr.strip()


def _path_entries() -> list[str]:
    return [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry.strip()]


def _find_cudnn_dll() -> str | None:
    for entry in _path_entries():
        candidate = Path(entry) / REQUIRED_CUDNN_DLL
        if candidate.exists():
            return str(candidate)
    return None


def _find_dll(dll_name: str) -> str | None:
    for entry in _path_entries():
        candidate = Path(entry) / dll_name
        if candidate.exists():
            return str(candidate)
    return None


def run_windows_gpu_check() -> WindowsGpuCheckResult:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    if sys.platform != "win32":
        errors.append("This checker is for native Windows only (sys.platform must be 'win32').")

    if sys.version_info[:2] != REQUIRED_PYTHON:
        errors.append("Python must be 3.10.x for native Windows TensorFlow GPU stack. " f"Found: {sys.version.split()[0]}")

    nvidia_smi = shutil.which("nvidia-smi")
    details["nvidia_smi_path"] = nvidia_smi
    if not nvidia_smi:
        errors.append("nvidia-smi is not available in PATH.")
        details["nvidia_smi_output"] = ""
    else:
        code, stdout, stderr = _run_command([nvidia_smi, "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"])
        details["nvidia_smi_output"] = stdout
        if code != 0:
            errors.append(f"nvidia-smi failed with exit code {code}: {stderr or stdout}")

    nvcc = shutil.which("nvcc")
    details["nvcc_path"] = nvcc
    if nvcc:
        code, stdout, stderr = _run_command([nvcc, "--version"])
        nvcc_text = stdout or stderr
        details["nvcc_version_output"] = nvcc_text
        if code != 0:
            warnings.append(f"nvcc --version exited with code {code}.")
        if REQUIRED_CUDA_VERSION_HINT not in nvcc_text:
            warnings.append("nvcc output does not mention CUDA 11.2. TensorFlow 2.10 native Windows GPU expects CUDA 11.2.")
    else:
        details["nvcc_version_output"] = ""
        warnings.append("nvcc is not available in PATH; CUDA toolkit version cannot be confirmed via nvcc.")

    cuda_path_entries = [entry for entry in _path_entries() if "cuda" in entry.lower() or "nvidia gpu computing toolkit" in entry.lower()]
    details["cuda_path_entries"] = cuda_path_entries
    if not cuda_path_entries:
        errors.append("No CUDA-related directories found in PATH.")
    cuda_112_entries = [entry for entry in cuda_path_entries if "v11.2" in entry.lower()]
    details["cuda_11_2_path_entries"] = cuda_112_entries
    if not cuda_112_entries:
        errors.append("CUDA 11.2 path entries were not found in PATH. Native Windows TensorFlow 2.10 requires CUDA 11.2.")

    cudnn_library = ctypes.util.find_library("cudnn64_8")
    cudnn_dll_path = _find_cudnn_dll()
    details["cudnn_find_library"] = cudnn_library
    details["cudnn_dll_path"] = cudnn_dll_path
    if not cudnn_library and not cudnn_dll_path:
        errors.append("cuDNN 8.1 DLL was not found. Expected to find cudnn64_8.dll in PATH/CUDA directories.")

    required_cuda_dll_locations = {dll_name: _find_dll(dll_name) for dll_name in REQUIRED_CUDA_DLLS}
    details["required_cuda_dlls"] = required_cuda_dll_locations
    missing_cuda_dlls = [name for name, location in required_cuda_dll_locations.items() if location is None]
    if missing_cuda_dlls:
        errors.append("Required CUDA runtime DLLs are missing from PATH: " + ", ".join(missing_cuda_dlls) + ". Install CUDA 11.2 and ensure CUDA bin directory is in PATH.")

    try:
        import tensorflow as tf
    except Exception as error:
        details["tensorflow_imported"] = False
        details["tensorflow_error"] = str(error)
        errors.append(f"TensorFlow import failed: {error}")
    else:
        tf_version = str(getattr(tf, "__version__", "unknown"))
        details["tensorflow_imported"] = True
        details["tensorflow_version"] = tf_version
        details["tensorflow_built_with_cuda"] = bool(tf.test.is_built_with_cuda())
        physical_gpus = tuple(str(device) for device in tf.config.list_physical_devices("GPU"))
        details["tensorflow_physical_gpus"] = list(physical_gpus)

        tf_major_minor = _parse_major_minor(tf_version)
        if tf_major_minor is None:
            errors.append(f"Could not parse TensorFlow version: {tf_version}")
        elif tf_major_minor > (2, 10):
            errors.append("TensorFlow version is unsupported for native Windows GPU. " f"Found {tf_version}; expected 2.10.x.")
        elif not tf_version.startswith(WINDOWS_TF_VERSION_PREFIX):
            warnings.append(f"TensorFlow is {tf_version}; expected 2.10.x for the native Windows GPU path.")

        if not details["tensorflow_built_with_cuda"]:
            errors.append("TensorFlow is not built with CUDA (tf.test.is_built_with_cuda() is False).")

        if not physical_gpus:
            errors.append("TensorFlow cannot see any physical GPU devices.")

    return WindowsGpuCheckResult(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        details=details,
    )


def _format(result: WindowsGpuCheckResult) -> str:
    lines = [
        f"Windows GPU check: {'OK' if result.ok else 'FAILED'}",
        f"Platform: {result.details.get('platform')}",
        f"Python: {result.details.get('python_version')} ({result.details.get('python_executable')})",
        f"CUDA_VISIBLE_DEVICES: {json.dumps(result.details.get('cuda_visible_devices'))}",
        f"TensorFlow version: {result.details.get('tensorflow_version', 'unavailable')}",
        f"TensorFlow built with CUDA: {result.details.get('tensorflow_built_with_cuda')}",
        f"TensorFlow physical GPUs: {len(result.details.get('tensorflow_physical_gpus', []))}",
        f"nvidia-smi: {result.details.get('nvidia_smi_path')}",
        f"nvcc: {result.details.get('nvcc_path')}",
        f"cuDNN: {result.details.get('cudnn_dll_path') or result.details.get('cudnn_find_library')}",
    ]
    nvidia_output = str(result.details.get("nvidia_smi_output") or "").strip()
    if nvidia_output:
        lines.append("nvidia-smi output:")
        lines.extend(f"- {line}" for line in nvidia_output.splitlines())
    cuda_entries = result.details.get("cuda_path_entries", [])
    lines.append(f"CUDA PATH entries: {len(cuda_entries)}")
    lines.extend(f"- {entry}" for entry in cuda_entries)
    cuda_112_entries = result.details.get("cuda_11_2_path_entries", [])
    lines.append(f"CUDA 11.2 PATH entries: {len(cuda_112_entries)}")
    lines.extend(f"- {entry}" for entry in cuda_112_entries)
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate native Windows TensorFlow 2.10 CUDA GPU stack.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_windows_gpu_check()
    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "errors": list(result.errors),
                    "warnings": list(result.warnings),
                    "details": result.details,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_format(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
