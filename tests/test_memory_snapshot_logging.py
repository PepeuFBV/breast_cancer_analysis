from __future__ import annotations

from unittest.mock import patch

from pipeline.utils.memory import log_memory_snapshot


def test_memory_snapshot_handles_missing_gpu_and_tensorflow() -> None:
    with patch("pipeline.utils.memory.get_gpu_memory_info", return_value=None):
        with patch("pipeline.utils.memory.get_tf_memory_info", return_value=None):
            snapshot = log_memory_snapshot("test:missing-providers")

    assert snapshot["label"] == "test:missing-providers"
    assert "process_memory_mb" in snapshot
    assert "peak_process_memory_mb" in snapshot
    assert snapshot["gpu_memory"] is None
    assert snapshot["tf_memory"] is None


def test_memory_snapshot_gracefully_handles_provider_exceptions() -> None:
    with patch(
        "pipeline.utils.memory.get_gpu_memory_info",
        side_effect=RuntimeError("gpu provider unavailable"),
    ):
        with patch(
            "pipeline.utils.memory.get_tf_memory_info",
            side_effect=RuntimeError("tf provider unavailable"),
        ):
            snapshot = log_memory_snapshot("test:provider-errors")

    assert snapshot["label"] == "test:provider-errors"
    assert snapshot["gpu_memory"]["error"] == "gpu provider unavailable"
    assert snapshot["tf_memory"]["error"] == "tf provider unavailable"
