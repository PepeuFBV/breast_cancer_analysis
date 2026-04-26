from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorFlowGpuCheck:
    mode: str
    tensorflow_available: bool
    tensorflow_version: str | None
    built_with_cuda: bool | None
    physical_gpus: tuple[str, ...]
    logical_gpus: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _device_name(device: Any) -> str:
    return str(getattr(device, "name", device))


def check_tensorflow_gpu(
    *,
    require_gpu: bool = False,
    cpu_only: bool = False,
    tensorflow_module: Any | None = None,
) -> TensorFlowGpuCheck:
    if require_gpu and cpu_only:
        return TensorFlowGpuCheck(
            mode="invalid",
            tensorflow_available=False,
            tensorflow_version=None,
            built_with_cuda=None,
            physical_gpus=(),
            logical_gpus=(),
            warnings=(),
            errors=("--require-gpu cannot be combined with --cpu-only.",),
        )

    mode = "cpu-only" if cpu_only else "required" if require_gpu else "optional"
    warnings: list[str] = []
    errors: list[str] = []

    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        tf = tensorflow_module or importlib.import_module("tensorflow")
    except Exception as error:
        return TensorFlowGpuCheck(
            mode=mode,
            tensorflow_available=False,
            tensorflow_version=None,
            built_with_cuda=None,
            physical_gpus=(),
            logical_gpus=(),
            warnings=(),
            errors=(f"TensorFlow import failed: {error}",),
        )

    try:
        physical_gpus = tuple(
            _device_name(device) for device in tf.config.list_physical_devices("GPU")
        )
    except Exception as error:
        physical_gpus = ()
        errors.append(f"Could not list physical GPU devices: {error}")

    try:
        logical_gpus = tuple(
            _device_name(device) for device in tf.config.list_logical_devices("GPU")
        )
    except Exception as error:
        logical_gpus = ()
        warnings.append(f"Could not list logical GPU devices: {error}")

    built_with_cuda: bool | None
    try:
        built_with_cuda = bool(tf.test.is_built_with_cuda())
    except Exception as error:
        built_with_cuda = None
        warnings.append(f"Could not determine CUDA build support: {error}")

    if require_gpu and not physical_gpus:
        errors.append(
            "No TensorFlow GPU devices are visible. Check NVIDIA driver, CUDA/cuDNN "
            "compatibility, WSL GPU passthrough if applicable, and "
            "CUDA_VISIBLE_DEVICES."
        )
    elif not require_gpu and not cpu_only and not physical_gpus:
        warnings.append(
            "No TensorFlow GPU devices are visible; CPU execution is available."
        )
    elif cpu_only and physical_gpus:
        warnings.append(
            "GPU devices are still visible in CPU-only mode. Set CUDA_VISIBLE_DEVICES "
            "before importing TensorFlow."
        )

    return TensorFlowGpuCheck(
        mode=mode,
        tensorflow_available=True,
        tensorflow_version=str(getattr(tf, "__version__", "unknown")),
        built_with_cuda=built_with_cuda,
        physical_gpus=physical_gpus,
        logical_gpus=logical_gpus,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def result_to_dict(result: TensorFlowGpuCheck) -> dict[str, object]:
    return {
        "ok": result.ok,
        "mode": result.mode,
        "tensorflow_available": result.tensorflow_available,
        "tensorflow_version": result.tensorflow_version,
        "built_with_cuda": result.built_with_cuda,
        "physical_gpus": list(result.physical_gpus),
        "logical_gpus": list(result.logical_gpus),
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }


def format_result(result: TensorFlowGpuCheck) -> str:
    status = "OK" if result.ok else "FAILED"
    lines = [
        f"TensorFlow GPU check: {status}",
        f"Mode: {result.mode}",
        f"TensorFlow available: {result.tensorflow_available}",
        f"TensorFlow version: {result.tensorflow_version or 'unavailable'}",
        f"Built with CUDA: {result.built_with_cuda}",
        f"Physical GPUs: {len(result.physical_gpus)}",
    ]
    lines.extend(f"- {gpu}" for gpu in result.physical_gpus)
    lines.append(f"Logical GPUs: {len(result.logical_gpus)}")
    lines.extend(f"- {gpu}" for gpu in result.logical_gpus)
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check TensorFlow import and GPU visibility."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if TensorFlow cannot see at least one GPU.",
    )
    mode.add_argument(
        "--cpu-only",
        action="store_true",
        help="Hide CUDA devices before importing TensorFlow and validate CPU mode.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_tensorflow_gpu(
        require_gpu=args.require_gpu,
        cpu_only=args.cpu_only,
    )
    if args.json:
        print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    else:
        print(format_result(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
