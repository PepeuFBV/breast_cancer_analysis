from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.gpu_env import bootstrap_tensorflow_runtime_env  # noqa: E402
from pipeline.utils.runtime_device import check_runtime_device  # noqa: E402


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
    device = "cpu" if cpu_only else "gpu" if require_gpu else "auto"
    resolved_tensorflow = tensorflow_module
    if resolved_tensorflow is None:
        bootstrap_tensorflow_runtime_env()
        try:
            resolved_tensorflow = importlib.import_module("tensorflow")
        except Exception as error:
            extra_help = ""
            if isinstance(error, ModuleNotFoundError) and getattr(error, "name", None) == "tensorflow":
                extra_help = " Install the project environment with " "`python3 scripts/bootstrap_env.py`."
            return TensorFlowGpuCheck(
                mode=mode,
                tensorflow_available=False,
                tensorflow_version=None,
                built_with_cuda=None,
                physical_gpus=(),
                logical_gpus=(),
                warnings=(),
                errors=(f"TensorFlow import failed: {error}.{extra_help}",),
            )
    runtime_result = check_runtime_device(
        device=device,
        require_gpu=require_gpu,
        tensorflow_module=resolved_tensorflow,
    )

    return TensorFlowGpuCheck(
        mode=mode,
        tensorflow_available=runtime_result.tensorflow_available,
        tensorflow_version=runtime_result.tensorflow_version,
        built_with_cuda=runtime_result.built_with_cuda,
        physical_gpus=runtime_result.physical_gpus,
        logical_gpus=runtime_result.logical_gpus,
        warnings=runtime_result.warnings,
        errors=runtime_result.errors,
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
    parser = argparse.ArgumentParser(description="Check TensorFlow import and GPU visibility.")
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
