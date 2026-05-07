from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any

from pipeline.utils.gpu_env import WINDOWS_NATIVE_TF_GPU_MAX_VERSION, bootstrap_tensorflow_runtime_env, is_native_windows


@dataclass(frozen=True)
class RuntimeDeviceCheck:
    requested_device: str
    selected_device: str
    tensorflow_available: bool
    tensorflow_version: str | None
    built_with_cuda: bool | None
    physical_gpus: tuple[str, ...]
    logical_gpus: tuple[str, ...]
    memory_growth_enabled: bool | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _device_name(device: Any) -> str:
    return str(getattr(device, "name", device))


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


def check_runtime_device(
    *,
    device: str,
    require_gpu: bool = False,
    tensorflow_module: Any | None = None,
) -> RuntimeDeviceCheck:
    warnings: list[str] = []
    errors: list[str] = []
    memory_growth_enabled: bool | None = None

    if device not in {"auto", "cpu", "gpu"}:
        return RuntimeDeviceCheck(
            requested_device=device,
            selected_device="unknown",
            tensorflow_available=False,
            tensorflow_version=None,
            built_with_cuda=None,
            physical_gpus=(),
            logical_gpus=(),
            memory_growth_enabled=None,
            warnings=(),
            errors=(f"Unsupported device mode: {device}",),
        )

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        os.environ.pop("CUDA_VISIBLE_DEVICES")

    if tensorflow_module is None:
        bootstrap_tensorflow_runtime_env()

    try:
        tf = tensorflow_module or importlib.import_module("tensorflow")
    except Exception as error:
        extra_help = ""
        if isinstance(error, ModuleNotFoundError) and getattr(error, "name", None) == "tensorflow":
            extra_help = " Install the project environment with " "`python3 scripts/bootstrap_env.py`."
        return RuntimeDeviceCheck(
            requested_device=device,
            selected_device="cpu" if device == "cpu" else "unavailable",
            tensorflow_available=False,
            tensorflow_version=None,
            built_with_cuda=None,
            physical_gpus=(),
            logical_gpus=(),
            memory_growth_enabled=None,
            warnings=(),
            errors=(f"TensorFlow import failed: {error}.{extra_help}",),
        )

    try:
        physical_devices = tuple(tf.config.list_physical_devices("GPU"))
        physical_gpus = tuple(_device_name(gpu) for gpu in physical_devices)
    except Exception as error:
        physical_devices = ()
        physical_gpus = ()
        errors.append(f"Could not list physical GPU devices: {error}")

    try:
        built_with_cuda = bool(tf.test.is_built_with_cuda())
    except Exception as error:
        built_with_cuda = None
        warnings.append(f"Could not determine CUDA build support: {error}")

    tensorflow_version = str(getattr(tf, "__version__", "unknown"))
    tensorflow_major_minor = _parse_major_minor(tensorflow_version)
    on_native_windows = is_native_windows() and not sys.platform.startswith("linux")
    windows_gpu_supported_tf = tensorflow_major_minor is not None and tensorflow_major_minor <= WINDOWS_NATIVE_TF_GPU_MAX_VERSION
    if on_native_windows and device != "cpu":
        if tensorflow_major_minor is None:
            warnings.append("Could not parse TensorFlow version on native Windows; native CUDA GPU support requires TensorFlow 2.10.x.")
        elif not windows_gpu_supported_tf:
            message = "Unsupported native Windows GPU stack: TensorFlow " f"{tensorflow_version} is installed, but native CUDA GPU is supported only on TensorFlow 2.10.x " "(with Python 3.10, CUDA 11.2, cuDNN 8.1)."
            if device == "gpu" or require_gpu:
                errors.append(message)
            else:
                warnings.append(message)

    if require_gpu and device == "cpu":
        errors.append("--require-gpu cannot be combined with --device cpu.")

    if device == "cpu":
        selected_device = "cpu"
    elif physical_devices:
        selected_device = "gpu"
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            memory_growth_enabled = True
        except Exception as error:
            memory_growth_enabled = False
            warnings.append(f"Could not enable GPU memory growth: {error}")
    else:
        selected_device = "cpu"
        warning = "No TensorFlow GPU devices are visible; CPU execution is available."
        if device == "gpu" and require_gpu:
            if on_native_windows and windows_gpu_supported_tf:
                errors.append("No TensorFlow GPU devices are visible on native Windows with TensorFlow 2.10.x. " "Verify Python 3.10, CUDA 11.2, cuDNN 8.1, CUDA bin paths in PATH, and CUDA_VISIBLE_DEVICES.")
            else:
                errors.append("No TensorFlow GPU devices are visible. Check NVIDIA driver, " "CUDA/cuDNN compatibility, WSL GPU passthrough if applicable, " "and CUDA_VISIBLE_DEVICES.")
        else:
            warnings.append(warning)

    try:
        logical_gpus = tuple(_device_name(gpu) for gpu in tf.config.list_logical_devices("GPU"))
    except Exception as error:
        logical_gpus = ()
        warnings.append(f"Could not list logical GPU devices: {error}")

    if device == "cpu" and logical_gpus:
        warnings.append("GPU devices are still visible in CPU mode. Set " "CUDA_VISIBLE_DEVICES before importing TensorFlow.")

    return RuntimeDeviceCheck(
        requested_device=device,
        selected_device=selected_device,
        tensorflow_available=True,
        tensorflow_version=tensorflow_version,
        built_with_cuda=built_with_cuda,
        physical_gpus=physical_gpus,
        logical_gpus=logical_gpus,
        memory_growth_enabled=memory_growth_enabled,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
