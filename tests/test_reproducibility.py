from __future__ import annotations

from pipeline.utils.reproducibility import _should_enable_tensorflow_determinism


def test_tensorflow_determinism_defaults_to_disabled_for_gpu_auto(monkeypatch) -> None:
    monkeypatch.delenv("BREAST_CANCER_ANALYSIS_ENABLE_TF_DETERMINISM", raising=False)
    monkeypatch.delenv("BREAST_CANCER_ANALYSIS_REQUESTED_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert _should_enable_tensorflow_determinism() is False


def test_tensorflow_determinism_enables_for_cpu_mode(monkeypatch) -> None:
    monkeypatch.delenv("BREAST_CANCER_ANALYSIS_ENABLE_TF_DETERMINISM", raising=False)
    monkeypatch.setenv("BREAST_CANCER_ANALYSIS_REQUESTED_DEVICE", "cpu")

    assert _should_enable_tensorflow_determinism() is True


def test_tensorflow_determinism_honors_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("BREAST_CANCER_ANALYSIS_ENABLE_TF_DETERMINISM", "1")
    monkeypatch.setenv("BREAST_CANCER_ANALYSIS_REQUESTED_DEVICE", "gpu")

    assert _should_enable_tensorflow_determinism() is True
