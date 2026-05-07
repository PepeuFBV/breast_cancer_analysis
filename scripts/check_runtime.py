from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.runtime_device import (  # noqa: E402
    RuntimeDeviceCheck,
    check_runtime_device,
)


def result_to_dict(result: RuntimeDeviceCheck) -> dict[str, object]:
    return {
        "ok": result.ok,
        "requested_device": result.requested_device,
        "selected_device": result.selected_device,
        "tensorflow_available": result.tensorflow_available,
        "tensorflow_version": result.tensorflow_version,
        "built_with_cuda": result.built_with_cuda,
        "physical_gpus": list(result.physical_gpus),
        "logical_gpus": list(result.logical_gpus),
        "memory_growth_enabled": result.memory_growth_enabled,
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }


def format_result(result: RuntimeDeviceCheck) -> str:
    status = "OK" if result.ok else "FAILED"
    lines = [
        f"Runtime device check: {status}",
        f"Requested device: {result.requested_device}",
        f"Selected device: {result.selected_device}",
        f"TensorFlow available: {result.tensorflow_available}",
        f"TensorFlow version: {result.tensorflow_version or 'unavailable'}",
        f"Built with CUDA: {result.built_with_cuda}",
        f"Memory growth enabled: {result.memory_growth_enabled}",
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
    parser = argparse.ArgumentParser(description="Check TensorFlow import and runtime device selection.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Runtime device policy to validate.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if the selected runtime cannot use TensorFlow on GPU.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_runtime_device(
        device=args.device,
        require_gpu=args.require_gpu,
    )
    if args.json:
        print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    else:
        print(format_result(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
