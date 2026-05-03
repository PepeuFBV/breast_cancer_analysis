from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.utils.memory import get_gpu_memory_info
from pipeline.utils.runtime_limits import (
    CpuExecutionLimits,
    apply_cpu_runtime_limits,
    current_cpu_thread_env,
    validate_cpu_execution_limits,
)

_WSL_NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")


@dataclass(frozen=True)
class RuntimeProbeResult:
    requested_device: str
    effective_device: str
    python_executable: str
    tensorflow_imported: bool
    tensorflow_version: str | None
    cuda_visible_devices: str | None
    physical_gpu_devices: tuple[str, ...]
    logical_gpu_devices: tuple[str, ...]
    tensorflow_visible_devices: tuple[str, ...]
    nvidia_smi_available: bool
    nvidia_smi_command: tuple[str, ...]
    gpu_memory_summary: dict[str, Any] | None
    effective_cpu_thread_env: dict[str, str | None]
    tensorflow_tiny_gpu_op: bool | None
    tensorflow_tiny_gpu_op_device: str | None
    gpu_used: bool
    opencv_threads: int | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _nvidia_smi_command() -> list[str] | None:
    if _WSL_NVIDIA_SMI.exists():
        return [str(_WSL_NVIDIA_SMI)]
    executable = shutil.which("nvidia-smi")
    if executable:
        return [executable]
    return None


def _device_name(device: Any) -> str:
    return str(getattr(device, "name", device))


def _resolve_effective_device(
    requested_device: str,
    *,
    logical_gpu_devices: tuple[str, ...],
    tiny_gpu_op_ok: bool | None,
) -> str:
    if requested_device == "cpu":
        return "cpu"
    if tiny_gpu_op_ok:
        return "gpu"
    if logical_gpu_devices:
        return "gpu"
    if requested_device == "gpu":
        return "cpu"
    return "cpu"


def _run_tiny_gpu_op(tf: Any) -> tuple[bool | None, str | None, str | None]:
    logical_gpus = tuple(_device_name(device) for device in tf.config.list_logical_devices("GPU"))
    if not logical_gpus:
        return False, None, None

    try:
        with tf.device("/GPU:0"):
            value = tf.constant([1.0], dtype=tf.float32) + tf.constant([2.0], dtype=tf.float32)
            try:
                value.numpy()
            except Exception:
                pass
        resolved_device = str(getattr(value, "device", None) or "")
        gpu_used = "GPU" in resolved_device.upper() or "XLA_GPU" in resolved_device.upper()
        return gpu_used, resolved_device or None, None if gpu_used else "Tiny TensorFlow op did not execute on GPU."
    except Exception as error:
        return False, None, str(error)


def collect_runtime_probe(
    *,
    device: str,
    cpu_execution_limits: CpuExecutionLimits | None = None,
    tensorflow_module: Any | None = None,
) -> RuntimeProbeResult:
    if device not in {"auto", "cpu", "gpu"}:
        return RuntimeProbeResult(
            requested_device=device,
            effective_device="unknown",
            python_executable=sys.executable,
            tensorflow_imported=False,
            tensorflow_version=None,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            physical_gpu_devices=(),
            logical_gpu_devices=(),
            tensorflow_visible_devices=(),
            nvidia_smi_available=False,
            nvidia_smi_command=(),
            gpu_memory_summary=None,
            effective_cpu_thread_env=current_cpu_thread_env(),
            tensorflow_tiny_gpu_op=None,
            tensorflow_tiny_gpu_op_device=None,
            gpu_used=False,
            opencv_threads=None,
            warnings=(),
            errors=(f"Unsupported device mode: {device}",),
        )

    warnings: list[str] = []
    errors: list[str] = []
    opencv_threads: int | None = None

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        limits = validate_cpu_execution_limits(cpu_execution_limits or CpuExecutionLimits())
        applied_limits = apply_cpu_runtime_limits(limits)
        opencv_threads = applied_limits.get("opencv_threads")
        warnings.extend(str(item) for item in applied_limits.get("warnings", []))

    tensorflow_imported = False
    tensorflow_version: str | None = None
    physical_gpu_devices: tuple[str, ...] = ()
    logical_gpu_devices: tuple[str, ...] = ()
    tiny_gpu_op_ok: bool | None = None
    tiny_gpu_op_device: str | None = None

    try:
        tf = tensorflow_module or importlib.import_module("tensorflow")
        tensorflow_imported = True
        tensorflow_version = str(getattr(tf, "__version__", "unknown"))
        physical_gpu_devices = tuple(_device_name(device) for device in tf.config.list_physical_devices("GPU"))
        logical_gpu_devices = tuple(_device_name(device) for device in tf.config.list_logical_devices("GPU"))
        if device != "cpu":
            tiny_gpu_op_ok, tiny_gpu_op_device, tiny_gpu_op_error = _run_tiny_gpu_op(tf)
            if tiny_gpu_op_error:
                warnings.append(tiny_gpu_op_error)
    except Exception as error:
        errors.append(f"TensorFlow import failed: {error}")

    if device == "gpu":
        if not tensorflow_imported:
            pass
        elif not logical_gpu_devices:
            errors.append("No TensorFlow GPU devices are visible.")
        elif not tiny_gpu_op_ok:
            errors.append("TensorFlow could not run a tiny operation on GPU.")
    elif device == "auto" and tensorflow_imported and logical_gpu_devices and not tiny_gpu_op_ok:
        warnings.append("TensorFlow detected GPU devices, but the tiny GPU operation failed.")

    if device == "cpu" and logical_gpu_devices:
        warnings.append("GPU devices are still visible in CPU mode. CUDA_VISIBLE_DEVICES may have been applied too late.")

    nvidia_smi_command = _nvidia_smi_command() or []
    gpu_memory_summary = get_gpu_memory_info() if nvidia_smi_command else None
    effective_device = _resolve_effective_device(
        device,
        logical_gpu_devices=logical_gpu_devices,
        tiny_gpu_op_ok=tiny_gpu_op_ok,
    )

    return RuntimeProbeResult(
        requested_device=device,
        effective_device=effective_device,
        python_executable=sys.executable,
        tensorflow_imported=tensorflow_imported,
        tensorflow_version=tensorflow_version,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        physical_gpu_devices=physical_gpu_devices,
        logical_gpu_devices=logical_gpu_devices,
        tensorflow_visible_devices=logical_gpu_devices,
        nvidia_smi_available=bool(nvidia_smi_command),
        nvidia_smi_command=tuple(nvidia_smi_command),
        gpu_memory_summary=gpu_memory_summary,
        effective_cpu_thread_env=current_cpu_thread_env(),
        tensorflow_tiny_gpu_op=tiny_gpu_op_ok,
        tensorflow_tiny_gpu_op_device=tiny_gpu_op_device,
        gpu_used=bool(tiny_gpu_op_ok),
        opencv_threads=opencv_threads,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def runtime_probe_to_dict(result: RuntimeProbeResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "requested_device": result.requested_device,
        "effective_device": result.effective_device,
        "python_executable": result.python_executable,
        "tensorflow_imported": result.tensorflow_imported,
        "tensorflow_version": result.tensorflow_version,
        "cuda_visible_devices": result.cuda_visible_devices,
        "physical_gpu_devices": list(result.physical_gpu_devices),
        "logical_gpu_devices": list(result.logical_gpu_devices),
        "tensorflow_visible_devices": list(result.tensorflow_visible_devices),
        "nvidia_smi_available": result.nvidia_smi_available,
        "nvidia_smi_command": list(result.nvidia_smi_command),
        "gpu_memory_summary": result.gpu_memory_summary,
        "effective_cpu_thread_env": result.effective_cpu_thread_env,
        "tensorflow_tiny_gpu_op": result.tensorflow_tiny_gpu_op,
        "tensorflow_tiny_gpu_op_device": result.tensorflow_tiny_gpu_op_device,
        "gpu_used": result.gpu_used,
        "opencv_threads": result.opencv_threads,
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }


def format_runtime_probe(result: RuntimeProbeResult) -> str:
    payload = runtime_probe_to_dict(result)
    lines = [
        f"Runtime probe: {'OK' if result.ok else 'FAILED'}",
        f"Requested device: {result.requested_device}",
        f"Effective device: {result.effective_device}",
        f"Python executable: {result.python_executable}",
        f"TensorFlow imported: {result.tensorflow_imported}",
        f"TensorFlow version: {result.tensorflow_version or 'unavailable'}",
        f"CUDA_VISIBLE_DEVICES: {json.dumps(result.cuda_visible_devices)}",
        f"nvidia-smi available: {result.nvidia_smi_available}",
        f"Physical GPU devices: {len(result.physical_gpu_devices)}",
    ]
    lines.extend(f"- {device_name}" for device_name in result.physical_gpu_devices)
    lines.append(f"Logical GPU devices: {len(result.logical_gpu_devices)}")
    lines.extend(f"- {device_name}" for device_name in result.logical_gpu_devices)
    lines.append(f"Tiny TensorFlow GPU op: {result.tensorflow_tiny_gpu_op}")
    lines.append(f"Tiny TensorFlow GPU op device: {result.tensorflow_tiny_gpu_op_device}")
    lines.append("Effective CPU thread env:")
    lines.extend(f"- {key}={value}" for key, value in sorted(payload["effective_cpu_thread_env"].items()))
    lines.append(f"OpenCV threads: {result.opencv_threads}")
    lines.append(f"GPU memory summary: {json.dumps(result.gpu_memory_summary, sort_keys=True)}")
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines)
