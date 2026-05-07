from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLayoutValidation:
    raw_data_dir: Path
    metadata_path: Path
    dicom_dir: Path
    dicom_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def inspect_dataset_layout(
    raw_data_dir: str | Path,
    *,
    metadata_filename: str = "INbreast.csv",
    dicom_subdir: str = "AllDICOMs",
    min_dicoms: int = 1,
) -> DatasetLayoutValidation:
    resolved_raw_data_dir = Path(raw_data_dir).expanduser()
    metadata_path = resolved_raw_data_dir / metadata_filename
    dicom_dir = resolved_raw_data_dir / dicom_subdir

    errors: list[str] = []
    warnings: list[str] = []
    dicom_count = 0

    if not resolved_raw_data_dir.exists():
        errors.append(f"dataset directory is missing: {resolved_raw_data_dir}")
    elif not resolved_raw_data_dir.is_dir():
        errors.append(f"dataset path is not a directory: {resolved_raw_data_dir}")

    if not metadata_path.exists():
        errors.append(f"metadata CSV is missing: {metadata_path}")
    elif not metadata_path.is_file():
        errors.append(f"metadata path is not a file: {metadata_path}")

    if not dicom_dir.exists():
        errors.append(f"DICOM directory is missing: {dicom_dir}")
    elif not dicom_dir.is_dir():
        errors.append(f"DICOM path is not a directory: {dicom_dir}")
    else:
        dicom_count = sum(1 for path in dicom_dir.glob("*.dcm") if path.is_file())
        if dicom_count < min_dicoms:
            errors.append(f"expected at least {min_dicoms} .dcm file(s) under {dicom_dir}, " f"found {dicom_count}")

    if metadata_path.exists() and metadata_path.stat().st_size == 0:
        warnings.append(f"metadata CSV is empty: {metadata_path}")

    return DatasetLayoutValidation(
        raw_data_dir=resolved_raw_data_dir,
        metadata_path=metadata_path,
        dicom_dir=dicom_dir,
        dicom_count=dicom_count,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def format_dataset_layout_report(result: DatasetLayoutValidation) -> str:
    status = "OK" if result.ok else "FAILED"
    lines = [
        f"Dataset layout: {status}",
        f"Raw data dir: {result.raw_data_dir}",
        f"Metadata CSV: {result.metadata_path}",
        f"DICOM dir: {result.dicom_dir}",
        f"DICOM files: {result.dicom_count}",
    ]
    if result.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in result.errors)
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    if not result.ok:
        lines.extend(
            [
                "Expected layout:",
                f"- {result.raw_data_dir / 'INbreast.csv'}",
                f"- {result.raw_data_dir / 'AllDICOMs' / '*.dcm'}",
            ]
        )
    return "\n".join(lines)


def validate_dataset_layout(
    raw_data_dir: str | Path,
    *,
    metadata_filename: str = "INbreast.csv",
    dicom_subdir: str = "AllDICOMs",
    min_dicoms: int = 1,
) -> DatasetLayoutValidation:
    result = inspect_dataset_layout(
        raw_data_dir,
        metadata_filename=metadata_filename,
        dicom_subdir=dicom_subdir,
        min_dicoms=min_dicoms,
    )
    if not result.ok:
        raise FileNotFoundError(format_dataset_layout_report(result))
    return result
