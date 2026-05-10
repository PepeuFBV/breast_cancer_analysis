from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import (
    check_environment,
    check_gpu,
    smoke_run,
    validate_dataset,
    validate_long_runner,
)

pytestmark = pytest.mark.unit


def _write_minimal_dataset(root: Path) -> Path:
    raw_dir = root / "INbreast Release 1.0"
    dicom_dir = raw_dir / "AllDICOMs"
    dicom_dir.mkdir(parents=True)
    (raw_dir / "INbreast.csv").write_text("File Name;Bi-Rads\n1;1\n", encoding="utf-8")
    (dicom_dir / "sample.dcm").write_bytes(b"fake")
    return raw_dir


class _FakeTensorFlowConfig:
    def __init__(self, physical_gpus=None, logical_gpus=None) -> None:
        self.physical_gpus = physical_gpus or []
        self.logical_gpus = logical_gpus or []

    def list_physical_devices(self, device_type: str):
        return self.physical_gpus if device_type == "GPU" else []

    def list_logical_devices(self, device_type: str):
        return self.logical_gpus if device_type == "GPU" else []


def _fake_tensorflow(*, physical_gpus=None, logical_gpus=None, cuda: bool = True):
    return SimpleNamespace(
        __version__="2.10.1",
        config=_FakeTensorFlowConfig(physical_gpus, logical_gpus),
        test=SimpleNamespace(is_built_with_cuda=lambda: cuda),
    )


@pytest.mark.gpu
def test_gpu_optional_warns_without_visible_devices() -> None:
    result = check_gpu.check_tensorflow_gpu(tensorflow_module=_fake_tensorflow(physical_gpus=[]))

    assert result.ok
    assert result.mode == "optional"
    assert "No TensorFlow GPU devices" in result.warnings[0]


@pytest.mark.gpu
def test_gpu_required_fails_without_visible_devices() -> None:
    result = check_gpu.check_tensorflow_gpu(
        require_gpu=True,
        tensorflow_module=_fake_tensorflow(physical_gpus=[]),
    )

    assert not result.ok
    assert result.mode == "required"
    assert "No TensorFlow GPU devices" in result.errors[0]


@pytest.mark.gpu
def test_gpu_required_accepts_mocked_gpu() -> None:
    gpu = SimpleNamespace(name="/physical_device:GPU:0")

    result = check_gpu.check_tensorflow_gpu(
        require_gpu=True,
        tensorflow_module=_fake_tensorflow(physical_gpus=[gpu], logical_gpus=[gpu]),
    )

    assert result.ok
    assert result.physical_gpus == ("/physical_device:GPU:0",)


def test_validate_dataset_script_reports_json_success(tmp_path, capsys) -> None:
    raw_dir = _write_minimal_dataset(tmp_path)

    exit_code = validate_dataset.main(
        [
            "--raw-data-dir",
            str(raw_dir),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["dicom_count"] == 1


def test_environment_check_reports_missing_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        check_environment,
        "_distribution_versions",
        lambda distributions: (
            {name: "test-version" for name in distributions},
            [],
        ),
    )

    result = check_environment.run_environment_check(
        raw_data_dir=tmp_path / "missing-data",
        artifacts_dir=tmp_path / "artifacts",
        skip_tensorflow=True,
    )

    assert not result.ok
    assert any("dataset directory is missing" in error for error in result.errors)
    assert (tmp_path / "artifacts").exists()


def test_environment_check_bootstraps_runtime_env_before_tensorflow_import(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        check_environment,
        "_distribution_versions",
        lambda distributions: (
            {name: "test-version" for name in distributions},
            [],
        ),
    )

    bootstrap_calls: list[str] = []
    monkeypatch.setattr(
        check_environment,
        "bootstrap_tensorflow_runtime_env",
        lambda: bootstrap_calls.append("called"),
    )
    monkeypatch.setattr(
        check_environment.importlib,
        "import_module",
        lambda name: _fake_tensorflow(),
    )

    result = check_environment.run_environment_check(
        raw_data_dir=tmp_path / "missing-data",
        artifacts_dir=tmp_path / "artifacts",
        skip_dataset=True,
    )

    assert result.ok
    assert bootstrap_calls == ["called"]


def test_gpu_check_missing_tensorflow_points_to_bootstrap() -> None:
    def _raise_missing_tensorflow(_name: str):
        error = ModuleNotFoundError("No module named 'tensorflow'")
        error.name = "tensorflow"
        raise error

    with patch.object(
        check_gpu.importlib,
        "import_module",
        side_effect=_raise_missing_tensorflow,
    ):
        result = check_gpu.check_tensorflow_gpu()

    assert not result.ok
    assert "python scripts/bootstrap_env.py" in result.errors[0]


@pytest.mark.smoke
def test_smoke_run_uses_synthetic_data_and_writes_report(tmp_path) -> None:
    snapshot, report_path, records = smoke_run._run_smoke(
        config_path=None,
        artifacts_dir=tmp_path / "artifacts",
        work_dir=tmp_path / "work",
        task_count=2,
    )

    assert snapshot["counts"]["completed"] == 2
    assert snapshot["counts"]["failed"] == 0
    assert records == 2
    assert report_path.exists()


def test_validate_long_runner_smoke_config_exists() -> None:
    assert validate_long_runner.DEFAULT_CONFIG_PATH == "configs/experiment.smoke.json"
