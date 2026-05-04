from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import check_windows_gpu

pytestmark = pytest.mark.unit


def _fake_tensorflow(*, version: str, built_with_cuda: bool, gpu_count: int):
    gpus = [SimpleNamespace(name=f"/physical_device:GPU:{idx}") for idx in range(gpu_count)]
    return SimpleNamespace(
        __version__=version,
        test=SimpleNamespace(is_built_with_cuda=lambda: built_with_cuda),
        config=SimpleNamespace(
            list_physical_devices=lambda device_type: gpus if device_type == "GPU" else [],
        ),
    )


def test_windows_gpu_checker_fails_outside_windows(monkeypatch) -> None:
    monkeypatch.setattr(check_windows_gpu.sys, "platform", "linux")
    monkeypatch.setattr(check_windows_gpu.sys, "version_info", (3, 10, 13, "final", 0))

    result = check_windows_gpu.run_windows_gpu_check()

    assert not result.ok
    assert any("native Windows only" in error for error in result.errors)


def test_windows_gpu_checker_flags_tf211_as_unsupported(monkeypatch, tmp_path) -> None:
    fake_tf = _fake_tensorflow(version="2.11.0", built_with_cuda=False, gpu_count=0)
    monkeypatch.setattr(check_windows_gpu.sys, "platform", "win32")
    monkeypatch.setattr(check_windows_gpu.sys, "version_info", (3, 10, 11, "final", 0))
    monkeypatch.setattr(check_windows_gpu.sys, "version", "3.10.11 (main, Jan 1 2026, 00:00:00)")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setitem(check_windows_gpu.sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr(check_windows_gpu.shutil, "which", lambda name: None)
    monkeypatch.setattr(check_windows_gpu.ctypes.util, "find_library", lambda name: None)

    result = check_windows_gpu.run_windows_gpu_check()

    assert not result.ok
    assert any("unsupported for native Windows GPU" in error for error in result.errors)


def test_windows_gpu_checker_requires_cuda_112_path_entries(monkeypatch, tmp_path) -> None:
    fake_tf = _fake_tensorflow(version="2.10.1", built_with_cuda=True, gpu_count=1)
    monkeypatch.setattr(check_windows_gpu.sys, "platform", "win32")
    monkeypatch.setattr(check_windows_gpu.sys, "version_info", (3, 10, 11, "final", 0))
    monkeypatch.setenv("PATH", str(tmp_path / "CUDA" / "v11.8" / "bin"))
    monkeypatch.setitem(check_windows_gpu.sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr(check_windows_gpu.shutil, "which", lambda name: None)
    monkeypatch.setattr(check_windows_gpu.ctypes.util, "find_library", lambda name: None)

    result = check_windows_gpu.run_windows_gpu_check()

    assert not result.ok
    assert any("CUDA 11.2 path entries were not found" in error for error in result.errors)
