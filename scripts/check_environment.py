from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import load_experiment_config
from pipeline.data.validation import inspect_dataset_layout
from pipeline.utils.gpu_env import ensure_tensorflow_wsl_gpu_env

MIN_PYTHON = (3, 10)
MAX_PYTHON_EXCLUSIVE = (3, 13)

REQUIRED_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "keras",
    "tensorflow",
    "opencv-python",
    "albumentations",
    "pydicom",
    "scikit-learn",
    "streamlit",
)


@dataclass(frozen=True)
class EnvironmentCheckResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    details: dict[str, object]

    @property
    def ok(self) -> bool:
        return not self.errors


def _in_virtualenv() -> bool:
    return bool(os.environ.get("VIRTUAL_ENV")) or sys.prefix != sys.base_prefix


def _distribution_versions(
    distributions: tuple[str, ...],
) -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    return versions, missing


def _check_writable_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path,
        prefix=".write-check-",
        delete=True,
        encoding="utf-8",
    ) as handle:
        handle.write("ok")
        handle.flush()


def run_environment_check(
    *,
    config_path: str | Path | None = None,
    raw_data_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    require_venv: bool = False,
    skip_dataset: bool = False,
    skip_tensorflow: bool = False,
) -> EnvironmentCheckResult:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, object] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "virtualenv": _in_virtualenv(),
    }

    if not (MIN_PYTHON <= sys.version_info[:2] < MAX_PYTHON_EXCLUSIVE):
        errors.append(
            "Python must be >=3.10 and <3.13 for the validated TensorFlow stack; "
            f"found {sys.version.split()[0]}."
        )

    if require_venv and not _in_virtualenv():
        errors.append(
            "No active virtual environment detected. Create one with "
            "`python3 -m venv .venv` and activate it before installing."
        )
    elif not _in_virtualenv():
        warnings.append(
            "No active virtual environment detected; using a venv is recommended."
        )

    versions, missing = _distribution_versions(REQUIRED_DISTRIBUTIONS)
    details["packages"] = versions
    if missing:
        errors.append(
            "Missing required package(s): "
            f"{', '.join(missing)}. Install with "
            "`python3 scripts/bootstrap_env.py`."
        )

    if not skip_tensorflow and "tensorflow" not in missing:
        try:
            ensure_tensorflow_wsl_gpu_env()
            tf = importlib.import_module("tensorflow")
            details["tensorflow_version"] = getattr(tf, "__version__", "unknown")
            details["tensorflow_physical_gpus"] = [
                str(device) for device in tf.config.list_physical_devices("GPU")
            ]
        except Exception as error:
            errors.append(f"TensorFlow import failed: {error}")

    try:
        experiment_config = load_experiment_config(config_path)
        project_paths = experiment_config.resolve_project_paths(
            raw_data_dir=raw_data_dir,
            artifacts_dir=artifacts_dir,
        )
        details["raw_data_dir"] = str(project_paths.raw_data_dir)
        details["artifacts_dir"] = str(project_paths.artifacts_dir)
    except Exception as error:
        errors.append(f"Could not load experiment config: {error}")
        return EnvironmentCheckResult(
            errors=tuple(errors),
            warnings=tuple(warnings),
            details=details,
        )

    writable_dirs = (
        project_paths.artifacts_dir,
        project_paths.processed_dir,
        project_paths.history_dir,
        project_paths.predictions_dir,
        project_paths.reports_dir,
        project_paths.experiment_state_dir,
    )
    unwritable: list[str] = []
    for directory in writable_dirs:
        try:
            _check_writable_directory(directory)
        except OSError as error:
            unwritable.append(f"{directory}: {error}")
    if unwritable:
        errors.append("Artifact directory write check failed: " + "; ".join(unwritable))

    if not skip_dataset:
        dataset_result = inspect_dataset_layout(project_paths.raw_data_dir)
        details["dataset"] = {
            "ok": dataset_result.ok,
            "metadata_path": str(dataset_result.metadata_path),
            "dicom_dir": str(dataset_result.dicom_dir),
            "dicom_count": dataset_result.dicom_count,
            "errors": list(dataset_result.errors),
        }
        errors.extend(dataset_result.errors)

    return EnvironmentCheckResult(
        errors=tuple(errors),
        warnings=tuple(warnings),
        details=details,
    )


def result_to_dict(result: EnvironmentCheckResult) -> dict[str, object]:
    return {
        "ok": result.ok,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "details": result.details,
    }


def format_result(result: EnvironmentCheckResult) -> str:
    status = "OK" if result.ok else "FAILED"
    lines = [f"Environment check: {status}"]
    lines.append(f"Python: {result.details.get('python')}")
    lines.append(f"Executable: {result.details.get('executable')}")
    lines.append(f"Virtualenv: {result.details.get('virtualenv')}")
    lines.append(f"Artifacts dir: {result.details.get('artifacts_dir', 'unresolved')}")
    lines.append(f"Raw data dir: {result.details.get('raw_data_dir', 'unresolved')}")
    packages = result.details.get("packages", {})
    if isinstance(packages, dict):
        lines.append("Packages:")
        lines.extend(
            f"- {name}: {version}" for name, version in sorted(packages.items())
        )
    dataset = result.details.get("dataset")
    if isinstance(dataset, dict):
        lines.append(
            f"Dataset: {'OK' if dataset.get('ok') else 'FAILED'} "
            f"({dataset.get('dicom_count')} DICOM files)"
        )
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    if result.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local setup before running the project pipeline."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument(
        "--require-venv",
        action="store_true",
        help="Fail when not running from an activated virtual environment.",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip INbreast dataset layout validation.",
    )
    parser.add_argument(
        "--skip-tensorflow",
        action="store_true",
        help="Check installed distributions without importing TensorFlow.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_environment_check(
        config_path=args.config,
        raw_data_dir=args.raw_data_dir,
        artifacts_dir=args.artifacts_dir,
        require_venv=args.require_venv,
        skip_dataset=args.skip_dataset,
        skip_tensorflow=args.skip_tensorflow,
    )
    if args.json:
        print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    else:
        print(format_result(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
