from __future__ import annotations

import signal
from pathlib import Path

SIGINT = getattr(signal, "SIGINT", None)
SIGTERM = getattr(signal, "SIGTERM", None)
SIGKILL = getattr(signal, "SIGKILL", None)

OOM_MARKERS = (
    "resourceexhaustederror",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cublas alloc",
    "dnn library initialization failed",
    "failed to allocate memory",
    "oom",
    "out of memory",
)

MEMORY_PRESSURE_MARKERS = (
    "killed process",
    "cannot allocate memory",
    "oom killer",
    "out of memory",
)


def _read_text_tail(path: Path | None, *, max_bytes: int = 256_000) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="ignore")


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    haystack = text.lower()
    return any(marker in haystack for marker in markers)


def classify_task_failure(
    *,
    device: str | None,
    exit_code: int | None,
    timeout: bool,
    error_summary: str | None,
    task_events_path: Path | None = None,
    task_log_path: Path | None = None,
) -> str:
    interrupted_exit_codes = {130, 143}
    if SIGINT is not None:
        interrupted_exit_codes.add(-SIGINT)
    if SIGTERM is not None:
        interrupted_exit_codes.add(-SIGTERM)

    if timeout:
        return "timeout"
    if exit_code in interrupted_exit_codes:
        return "interrupted"

    base_text = error_summary or ""
    combined_text = "\n".join(
        [
            base_text,
            _read_text_tail(task_events_path),
            _read_text_tail(task_log_path),
        ]
    )
    # Always inspect aggregated logs in addition to `error_summary` because
    # subprocess failures may expose OOM markers only in stderr/task logs.
    has_oom_markers = _contains_any(combined_text, OOM_MARKERS)

    if has_oom_markers:
        if device == "gpu":
            return "gpu_oom"
        if device == "cpu":
            return "cpu_oom"
        return "oom"

    killed_exit_codes = {137}
    if SIGKILL is not None:
        killed_exit_codes.add(-SIGKILL)

    if exit_code in killed_exit_codes and _contains_any(combined_text, MEMORY_PRESSURE_MARKERS):
        if device == "gpu":
            return "gpu_oom"
        if device == "cpu":
            return "cpu_oom"
        return "oom"

    return "unknown_error"
