from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from pipeline.utils.thermal import collect_thermal_snapshot


def test_collect_thermal_snapshot_parses_gpu_and_cpu_metrics() -> None:
    fake_stdout = "\n".join(
        [
            "GPU 0, 78, 93",
            "GPU 1, 65, 50",
        ]
    )
    with (
        patch("pipeline.utils.thermal._nvidia_smi_command", return_value=["nvidia-smi"]),
        patch("pipeline.utils.thermal._read_cpu_temperature_celsius", return_value=82.0),
        patch("pipeline.utils.thermal._read_cpu_load_percent", return_value=76.5),
        patch(
            "pipeline.utils.thermal.subprocess.run",
            return_value=SimpleNamespace(stdout=fake_stdout),
        ),
    ):
        snapshot = collect_thermal_snapshot()

    assert snapshot.cpu_temperature_celsius == 82.0
    assert snapshot.cpu_load_percent == 76.5
    assert len(snapshot.gpu_samples) == 2
    assert snapshot.max_gpu_temperature_celsius == 78.0
    assert snapshot.max_gpu_utilization_percent == 93.0
    assert snapshot.warnings == ()


def test_collect_thermal_snapshot_handles_gpu_probe_errors() -> None:
    with (
        patch("pipeline.utils.thermal._nvidia_smi_command", return_value=["nvidia-smi"]),
        patch("pipeline.utils.thermal._read_cpu_temperature_celsius", return_value=None),
        patch("pipeline.utils.thermal._read_cpu_load_percent", return_value=None),
        patch(
            "pipeline.utils.thermal.subprocess.run",
            side_effect=RuntimeError("probe failed"),
        ),
    ):
        snapshot = collect_thermal_snapshot()

    assert snapshot.gpu_samples == ()
    assert snapshot.max_gpu_temperature_celsius is None
    assert snapshot.max_gpu_utilization_percent is None
    assert len(snapshot.warnings) == 1
    assert "probe failed" in snapshot.warnings[0]
