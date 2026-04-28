from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pipeline.utils.runtime_device import RuntimeDeviceCheck, check_runtime_device
from scripts import check_runtime


class _FakeExperimentalConfig:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def set_memory_growth(self, device, enabled: bool) -> None:
        self.calls.append((str(getattr(device, "name", device)), enabled))


class _FakeTensorFlowConfig:
    def __init__(self, physical_gpus=None, logical_gpus=None) -> None:
        self._physical_gpus = physical_gpus or []
        self._logical_gpus = logical_gpus or []
        self.experimental = _FakeExperimentalConfig()

    def list_physical_devices(self, device_type: str):
        return self._physical_gpus if device_type == "GPU" else []

    def list_logical_devices(self, device_type: str):
        return self._logical_gpus if device_type == "GPU" else []


def _fake_tensorflow(*, physical_gpus=None, logical_gpus=None, cuda: bool = True):
    return SimpleNamespace(
        __version__="test-tf",
        config=_FakeTensorFlowConfig(physical_gpus, logical_gpus),
        test=SimpleNamespace(is_built_with_cuda=lambda: cuda),
    )


@pytest.mark.gpu
def test_check_runtime_device_auto_selects_gpu_and_enables_growth() -> None:
    gpu = SimpleNamespace(name="/physical_device:GPU:0")
    tf = _fake_tensorflow(physical_gpus=[gpu], logical_gpus=[gpu])

    result = check_runtime_device(device="auto", tensorflow_module=tf)

    assert result.ok
    assert result.selected_device == "gpu"
    assert result.memory_growth_enabled is True
    assert tf.config.experimental.calls == [("/physical_device:GPU:0", True)]


@pytest.mark.gpu
def test_check_runtime_device_gpu_required_fails_without_gpu() -> None:
    result = check_runtime_device(
        device="gpu",
        require_gpu=True,
        tensorflow_module=_fake_tensorflow(physical_gpus=[]),
    )

    assert not result.ok
    assert result.selected_device == "cpu"
    assert "No TensorFlow GPU devices" in result.errors[0]


@pytest.mark.gpu
def test_check_runtime_device_cpu_mode_stays_on_cpu(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    result = check_runtime_device(
        device="cpu",
        tensorflow_module=_fake_tensorflow(physical_gpus=[]),
    )

    assert result.ok
    assert result.selected_device == "cpu"
    assert result.requested_device == "cpu"


def test_check_runtime_script_emits_json(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        check_runtime,
        "check_runtime_device",
        lambda **kwargs: RuntimeDeviceCheck(
            requested_device="auto",
            selected_device="cpu",
            tensorflow_available=True,
            tensorflow_version="test-tf",
            built_with_cuda=False,
            physical_gpus=(),
            logical_gpus=(),
            memory_growth_enabled=None,
            warnings=("CPU fallback",),
            errors=(),
        ),
    )

    exit_code = check_runtime.main(["--device", "auto", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["selected_device"] == "cpu"
    assert payload["warnings"] == ["CPU fallback"]
