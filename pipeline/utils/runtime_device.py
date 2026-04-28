from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

from pipeline.utils.gpu_env import ensure_tensorflow_wsl_gpu_env


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
        ensure_tensorflow_wsl_gpu_env()

    try:
        tf = tensorflow_module or importlib.import_module("tensorflow")
    except Exception as error:
        extra_help = ""
        if (
            isinstance(error, ModuleNotFoundError)
            and getattr(error, "name", None) == "tensorflow"
        ):
            extra_help = (
                " Install the project environment with "
                "`python3 scripts/bootstrap_env.py`."
            )
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
        warning = (
            "No TensorFlow GPU devices are visible; CPU execution is available."
        )
        if device == "gpu" and require_gpu:
            errors.append(
                "No TensorFlow GPU devices are visible. Check NVIDIA driver, "
                "CUDA/cuDNN compatibility, WSL GPU passthrough if applicable, "
                "and CUDA_VISIBLE_DEVICES."
            )
        else:
            warnings.append(warning)

    try:
        logical_gpus = tuple(
            _device_name(gpu) for gpu in tf.config.list_logical_devices("GPU")
        )
    except Exception as error:
        logical_gpus = ()
        warnings.append(f"Could not list logical GPU devices: {error}")

    if device == "cpu" and logical_gpus:
        warnings.append(
            "GPU devices are still visible in CPU mode. Set "
            "CUDA_VISIBLE_DEVICES before importing TensorFlow."
        )

    return RuntimeDeviceCheck(
        requested_device=device,
        selected_device=selected_device,
        tensorflow_available=True,
        tensorflow_version=str(getattr(tf, "__version__", "unknown")),
        built_with_cuda=built_with_cuda,
        physical_gpus=physical_gpus,
        logical_gpus=logical_gpus,
        memory_growth_enabled=memory_growth_enabled,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
