from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

CPU_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class CpuExecutionLimits:
    max_threads: int | None = None
    opencv_threads: int | None = None
    inter_op_threads: int | None = None
    intra_op_threads: int | None = None
    nice: int | None = None

    def is_configured(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_threads,
                self.opencv_threads,
                self.inter_op_threads,
                self.intra_op_threads,
                self.nice,
            )
        )

    def to_dict(self) -> dict[str, int | None]:
        return {
            "cpu_max_threads": self.max_threads,
            "cpu_opencv_threads": self.opencv_threads,
            "cpu_inter_op_threads": self.inter_op_threads,
            "cpu_intra_op_threads": self.intra_op_threads,
            "cpu_nice": self.nice,
        }

    def to_cli_args(self) -> list[str]:
        arguments: list[str] = []
        if self.max_threads is not None:
            arguments.extend(["--cpu-max-threads", str(self.max_threads)])
        if self.opencv_threads is not None:
            arguments.extend(["--cpu-opencv-threads", str(self.opencv_threads)])
        if self.inter_op_threads is not None:
            arguments.extend(["--cpu-inter-op-threads", str(self.inter_op_threads)])
        if self.intra_op_threads is not None:
            arguments.extend(["--cpu-intra-op-threads", str(self.intra_op_threads)])
        if self.nice is not None:
            arguments.extend(["--cpu-nice", str(self.nice)])
        return arguments


def _validate_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be > 0 when provided.")
    return resolved


def validate_cpu_execution_limits(limits: CpuExecutionLimits) -> CpuExecutionLimits:
    nice = limits.nice
    if nice is not None:
        nice = int(nice)
        if nice < -20 or nice > 19:
            raise ValueError("cpu_nice must be between -20 and 19 when provided.")

    return CpuExecutionLimits(
        max_threads=_validate_positive_int("cpu_max_threads", limits.max_threads),
        opencv_threads=_validate_positive_int("cpu_opencv_threads", limits.opencv_threads),
        inter_op_threads=_validate_positive_int("cpu_inter_op_threads", limits.inter_op_threads),
        intra_op_threads=_validate_positive_int("cpu_intra_op_threads", limits.intra_op_threads),
        nice=nice,
    )


def resolve_cpu_thread_env(limits: CpuExecutionLimits) -> dict[str, str]:
    validated = validate_cpu_execution_limits(limits)
    env: dict[str, str] = {}
    if validated.max_threads is not None:
        thread_value = str(validated.max_threads)
        for key in CPU_THREAD_ENV_KEYS:
            env[key] = thread_value

    intra_threads = validated.intra_op_threads
    if intra_threads is None:
        intra_threads = validated.max_threads
    if intra_threads is not None:
        env["TF_NUM_INTRAOP_THREADS"] = str(intra_threads)

    inter_threads = validated.inter_op_threads
    if inter_threads is None and validated.max_threads is not None:
        inter_threads = min(validated.max_threads, 1)
    if inter_threads is not None:
        env["TF_NUM_INTEROP_THREADS"] = str(inter_threads)

    return env


def current_cpu_thread_env() -> dict[str, str | None]:
    keys = [*CPU_THREAD_ENV_KEYS, "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]
    return {key: os.environ.get(key) for key in keys}


def apply_cpu_runtime_limits(
    limits: CpuExecutionLimits,
) -> dict[str, Any]:
    validated = validate_cpu_execution_limits(limits)
    resolved_thread_env = resolve_cpu_thread_env(validated)
    for key, value in resolved_thread_env.items():
        os.environ[key] = value

    applied: dict[str, Any] = {
        "requested_limits": validated.to_dict(),
        "effective_cpu_thread_env": current_cpu_thread_env(),
        "opencv_threads": None,
        "tensorflow_threading": {
            "intra_op_threads": None,
            "inter_op_threads": None,
        },
        "nice": {
            "requested": validated.nice,
            "applied": None,
            "supported": hasattr(os, "nice"),
        },
        "warnings": [],
    }

    if validated.opencv_threads is not None:
        try:
            import cv2

            cv2.setNumThreads(validated.opencv_threads)
            applied["opencv_threads"] = int(cv2.getNumThreads())
        except Exception as error:
            applied["warnings"].append(f"Could not configure OpenCV threads: {error}")

    if validated.intra_op_threads is not None or validated.inter_op_threads is not None:
        try:
            import tensorflow as tf

            if validated.intra_op_threads is not None:
                tf.config.threading.set_intra_op_parallelism_threads(validated.intra_op_threads)
                applied["tensorflow_threading"]["intra_op_threads"] = validated.intra_op_threads
            if validated.inter_op_threads is not None:
                tf.config.threading.set_inter_op_parallelism_threads(validated.inter_op_threads)
                applied["tensorflow_threading"]["inter_op_threads"] = validated.inter_op_threads
        except Exception as error:
            applied["warnings"].append(f"Could not configure TensorFlow CPU threads: {error}")

    if validated.nice is not None:
        if hasattr(os, "nice") and sys.platform.startswith("linux"):
            try:
                current_nice = int(os.nice(0))
                if validated.nice > current_nice:
                    applied["nice"]["applied"] = int(os.nice(validated.nice - current_nice))
                else:
                    applied["nice"]["applied"] = current_nice
            except OSError as error:
                applied["warnings"].append(f"Could not apply CPU nice level: {error}")
        elif validated.nice is not None:
            applied["warnings"].append("CPU nice is only applied on Linux hosts.")

    applied["warnings"] = list(applied["warnings"])
    return applied
