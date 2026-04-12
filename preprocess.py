from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.data.constants import (
    DEFAULT_AUGMENTATIONS_PER_IMAGE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SAMPLES_PER_CLASS,
    DEFAULT_TEST_SIZE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the INbreast dataset into reusable pipeline artifacts.")
    parser.add_argument("--raw-data-dir", default=None, help="Raw dataset directory. Defaults to data/INbreast Release 1.0/.")
    parser.add_argument("--artifacts-dir", default=None, help="Artifacts root directory. Defaults to artifacts/.")
    parser.add_argument("--augmentations-per-image", type=int, default=DEFAULT_AUGMENTATIONS_PER_IMAGE)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--resize-width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--resize-height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    return parser


def validate_raw_dataset_layout(raw_data_dir: Path) -> None:
    required_csv = raw_data_dir / "INbreast.csv"
    dicom_dir = raw_data_dir / "AllDICOMs"

    missing_items: list[str] = []
    if not raw_data_dir.exists():
        missing_items.append(f"dataset directory: {raw_data_dir}")
    if not required_csv.exists():
        missing_items.append(f"metadata CSV: {required_csv}")
    if not dicom_dir.exists():
        missing_items.append(f"DICOM directory: {dicom_dir}")
    elif not any(dicom_dir.glob("*.dcm")):
        missing_items.append(f"DICOM files under: {dicom_dir}")

    if not missing_items:
        return

    details = "\n".join(f"- missing {item}" for item in missing_items)
    raise FileNotFoundError(
        "INbreast dataset layout is incomplete.\n"
        f"{details}\n"
        "Expected structure:\n"
        f"- {raw_data_dir / 'INbreast.csv'}\n"
        f"- {raw_data_dir / 'AllDICOMs' / '*.dcm'}"
    )


def main() -> int:
    from pipeline.data.dataset import DatasetPreparationConfig, prepare_dataset
    from pipeline.utils.paths import build_project_paths

    args = build_parser().parse_args()
    paths = build_project_paths(args.raw_data_dir, args.artifacts_dir).ensure_artifact_dirs()
    validate_raw_dataset_layout(paths.raw_data_dir)
    config = DatasetPreparationConfig(
        raw_data_dir=paths.raw_data_dir,
        images_output_dir=paths.processed_images_dir,
        splits_output_dir=paths.processed_splits_dir,
        resize_dim=(args.resize_width, args.resize_height),
        augmentations_per_image=args.augmentations_per_image,
        samples_per_class=args.samples_per_class,
        test_size=args.test_size,
    )
    artifacts = prepare_dataset(config)
    print(f"Prepared dataset from: {paths.raw_data_dir}")
    print(f"Images directory: {artifacts.images_output_dir}")
    print(f"Train split: {artifacts.train_split_path}")
    print(f"Test split: {artifacts.test_split_path}")
    print(f"Filtered metadata rows: {artifacts.filtered_metadata_rows}")
    print(f"Generated images: {artifacts.total_generated_images}")
    print(f"Balanced images: {artifacts.balanced_images}")
    print(f"Train samples: {artifacts.train_samples}")
    print(f"Test samples: {artifacts.test_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
