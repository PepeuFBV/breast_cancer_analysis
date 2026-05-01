from __future__ import annotations

import gc
import logging
import resource
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pipeline.utils.gpu_env import clear_gpu_memory


def clear_ml_memory(*, clear_session: bool = True) -> None:
    """Release Python, Keras, and TensorFlow memory where possible."""

    if clear_session:
        clear_gpu_memory()
    for _ in range(2):
        gc.collect()


def get_process_memory_mb() -> float | None:
    """Return RSS memory for the current process when available."""

    try:
        import psutil  # type: ignore

        process = psutil.Process()
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

    if usage <= 0:
        return None

    if Path("/proc/self/status").exists():
        return round(usage / 1024, 2)
    return round(usage / (1024 * 1024), 2)


def get_peak_process_memory_mb() -> float | None:
    """Return peak RSS memory for the current process when available."""

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

    if usage <= 0:
        return None

    if Path("/proc/self/status").exists():
        return round(usage / 1024, 2)
    return round(usage / (1024 * 1024), 2)


def _nvidia_smi_command() -> list[str] | None:
    for candidate in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"):
        if candidate == "nvidia-smi" and shutil.which(candidate):
            return [candidate]
        if candidate != "nvidia-smi" and Path(candidate).exists():
            return [candidate]
    return None


def get_gpu_memory_info() -> dict[str, Any] | None:
    """Return GPU memory statistics when NVIDIA tooling is available."""

    command = _nvidia_smi_command()
    if command is None:
        return None

    query = [
        *command,
        "--query-gpu=name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            query,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as error:
        return {"error": str(error)}

    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, used_mb, total_mb = parts
        try:
            used_value = int(used_mb)
            total_value = int(total_mb)
        except ValueError:
            continue
        devices.append(
            {
                "name": name,
                "memory_used_mb": used_value,
                "memory_total_mb": total_value,
            }
        )

    return {"devices": devices}


def get_tf_memory_info() -> dict[str, Any] | None:
    """Return TensorFlow memory statistics when TensorFlow is available."""

    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return None

    try:
        devices = tf.config.list_physical_devices("GPU")
    except Exception as error:
        return {"error": str(error)}

    if not devices:
        return {"devices": []}

    resolved_devices: list[dict[str, Any]] = []
    for index, _ in enumerate(devices):
        logical_name = f"GPU:{index}"
        try:
            raw_stats = tf.config.experimental.get_memory_info(logical_name)
        except Exception as error:
            resolved_devices.append({"device": logical_name, "error": str(error)})
            continue

        current = raw_stats.get("current")
        peak = raw_stats.get("peak")
        resolved_devices.append(
            {
                "device": logical_name,
                "current_mb": (round(float(current) / (1024 * 1024), 2) if isinstance(current, (int, float)) else None),
                "peak_mb": (round(float(peak) / (1024 * 1024), 2) if isinstance(peak, (int, float)) else None),
            }
        )

    return {"devices": resolved_devices}


def log_memory_snapshot(
    label: str,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Capture a point-in-time memory snapshot and optionally log it."""

    try:
        gpu_memory = get_gpu_memory_info()
    except Exception as error:
        gpu_memory = {"error": str(error)}

    try:
        tf_memory = get_tf_memory_info()
    except Exception as error:
        tf_memory = {"error": str(error)}

    snapshot = {
        "label": label,
        "process_memory_mb": get_process_memory_mb(),
        "peak_process_memory_mb": get_peak_process_memory_mb(),
        "gpu_memory": gpu_memory,
        "tf_memory": tf_memory,
    }

    if logger is not None:
        message = f"[memory] {label}: process={snapshot['process_memory_mb']}MB, " f"peak={snapshot['peak_process_memory_mb']}MB"
        gpu_memory = snapshot["gpu_memory"]
        if isinstance(gpu_memory, dict) and gpu_memory.get("devices"):
            gpu_parts = [(f"{device['name']} " f"{device['memory_used_mb']}/{device['memory_total_mb']}MB") for device in gpu_memory["devices"]]
            message = f"{message}; gpu={'; '.join(gpu_parts)}"
        elif isinstance(gpu_memory, dict) and gpu_memory.get("error"):
            message = f"{message}; gpu_error={gpu_memory['error']}"
        logger.info(message)

    return snapshot
