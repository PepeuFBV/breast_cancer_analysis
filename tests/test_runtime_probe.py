from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from pipeline.utils.runtime_probe import collect_runtime_probe, runtime_probe_to_dict

pytestmark = pytest.mark.unit


class _FakeTensorFlowConfig:
    def __init__(self, *, physical_gpus=None, logical_gpus=None) -> None:
        self._physical_gpus = physical_gpus or []
        self._logical_gpus = logical_gpus or []

    def list_physical_devices(self, device_type: str):
        return self._physical_gpus if device_type == "GPU" else []

    def list_logical_devices(self, device_type: str):
        return self._logical_gpus if device_type == "GPU" else []


def _fake_tensorflow(*, physical_gpus=None, logical_gpus=None, version: str = "test-tf"):
    return SimpleNamespace(
        __version__=version,
        config=_FakeTensorFlowConfig(physical_gpus=physical_gpus, logical_gpus=logical_gpus),
        constant=lambda value, dtype=None: SimpleNamespace(device="/device:CPU:0", numpy=lambda: value),
        float32="float32",
        device=lambda name: nullcontext(),
    )


def test_collect_runtime_probe_cpu_mode_without_real_gpu(monkeypatch) -> None:
    monkeypatch.setattr("pipeline.utils.runtime_probe._nvidia_smi_command", lambda: None)

    result = collect_runtime_probe(
        device="cpu",
        tensorflow_module=_fake_tensorflow(),
    )

    assert result.ok
    assert result.requested_device == "cpu"
    assert result.effective_device == "cpu"
    assert result.cuda_visible_devices == "-1"
    assert result.logical_gpu_devices == ()
    assert result.gpu_used is False
    assert runtime_probe_to_dict(result)["effective_cpu_thread_env"]["OMP_NUM_THREADS"] is None


def test_collect_runtime_probe_handles_missing_tensorflow_gracefully(monkeypatch) -> None:
    monkeypatch.setattr("pipeline.utils.runtime_probe._nvidia_smi_command", lambda: None)
    monkeypatch.setattr(
        "pipeline.utils.runtime_probe.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'tensorflow'")),
    )

    result = collect_runtime_probe(device="gpu")

    assert not result.ok
    assert result.tensorflow_imported is False
    assert any("TensorFlow import failed" in error for error in result.errors)


def test_collect_runtime_probe_windows_tf211_reports_unsupported_native_gpu_stack(monkeypatch) -> None:
    monkeypatch.setattr("pipeline.utils.runtime_probe._nvidia_smi_command", lambda: None)
    monkeypatch.setattr("pipeline.utils.runtime_probe.is_native_windows", lambda: True)

    result = collect_runtime_probe(
        device="gpu",
        tensorflow_module=_fake_tensorflow(version="2.11.0"),
    )

    assert not result.ok
    assert any("Unsupported native Windows GPU stack" in error for error in result.errors)
