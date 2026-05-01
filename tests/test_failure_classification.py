from __future__ import annotations

import signal
from pathlib import Path

from pipeline.experiments.failures import classify_task_failure


def _write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_classify_timeout() -> None:
    assert (
        classify_task_failure(
            device="gpu",
            exit_code=None,
            timeout=True,
            error_summary=None,
        )
        == "timeout"
    )


def test_classify_interrupted() -> None:
    assert (
        classify_task_failure(
            device="gpu",
            exit_code=-signal.SIGTERM,
            timeout=False,
            error_summary=None,
        )
        == "interrupted"
    )


def test_classify_gpu_oom_from_error_summary() -> None:
    assert (
        classify_task_failure(
            device="gpu",
            exit_code=1,
            timeout=False,
            error_summary="tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM",
        )
        == "gpu_oom"
    )


def test_classify_cpu_oom_from_error_summary() -> None:
    assert (
        classify_task_failure(
            device="cpu",
            exit_code=1,
            timeout=False,
            error_summary="failed to allocate memory",
        )
        == "cpu_oom"
    )


def test_classify_oom_from_task_logs_even_with_generic_error_summary(tmp_path: Path) -> None:
    events_path = _write_text(
        tmp_path / "task.events.jsonl",
        '{"event":"task:failed","error_message":"CUDA_ERROR_OUT_OF_MEMORY"}\n',
    )
    task_log_path = _write_text(tmp_path / "task.log", "non-oom wrapper error")
    assert (
        classify_task_failure(
            device="gpu",
            exit_code=1,
            timeout=False,
            error_summary="subprocess exited with code 1",
            task_events_path=events_path,
            task_log_path=task_log_path,
        )
        == "gpu_oom"
    )


def test_classify_sigkill_with_memory_pressure_markers(tmp_path: Path) -> None:
    task_log_path = _write_text(tmp_path / "task.log", "Killed process due to OOM killer")
    assert (
        classify_task_failure(
            device="gpu",
            exit_code=137,
            timeout=False,
            error_summary=None,
            task_log_path=task_log_path,
        )
        == "gpu_oom"
    )


def test_classify_unknown_error() -> None:
    assert (
        classify_task_failure(
            device="gpu",
            exit_code=1,
            timeout=False,
            error_summary="ValueError: unsupported parameter",
        )
        == "unknown_error"
    )
