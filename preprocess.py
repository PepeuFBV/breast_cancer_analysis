from __future__ import annotations

import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the INbreast dataset into reusable pipeline artifacts.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an experiment JSON config. Defaults to configs/experiment.default.json.",
    )
    parser.add_argument("--raw-data-dir", default=None, help="Raw dataset directory. Defaults to data/INbreast Release 1.0/.")
    parser.add_argument("--artifacts-dir", default=None, help="Artifacts root directory. Defaults to artifacts/.")
    parser.add_argument("--augmentations-per-image", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
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


def build_dataset_config_from_args(args: argparse.Namespace):
    from pipeline.config import load_experiment_config

    experiment_config = load_experiment_config(args.config)
    project_paths = experiment_config.resolve_project_paths(
        raw_data_dir=args.raw_data_dir,
        artifacts_dir=args.artifacts_dir,
    ).ensure_artifact_dirs()
    resize_dim = None
    if args.resize_width is not None or args.resize_height is not None:
        resize_dim = (
            args.resize_width if args.resize_width is not None else experiment_config.preprocess.image_size[0],
            args.resize_height if args.resize_height is not None else experiment_config.preprocess.image_size[1],
        )
    return experiment_config.build_dataset_preparation_config(
        project_paths,
        image_size=resize_dim,
        augmentations_per_image=args.augmentations_per_image,
        samples_per_class=args.samples_per_class,
        test_size=args.test_size,
        random_state=args.random_state,
    )


def main() -> int:
    from pipeline.data.dataset import prepare_dataset
    from pipeline.utils.reproducibility import enforce_reproducibility

    args = build_parser().parse_args()
    config = build_dataset_config_from_args(args)
    validate_raw_dataset_layout(config.raw_data_dir)
    enforce_reproducibility(config.random_state)
    artifacts = prepare_dataset(config)
    print(f"Prepared dataset from: {config.raw_data_dir}")
    print(f"Images directory: {artifacts.images_output_dir}")
    print(f"Train split: {artifacts.train_split_path}")
    print(f"Test split: {artifacts.test_split_path}")
    print(f"Split manifest: {artifacts.split_manifest_path}")
    print(f"Filtered metadata rows: {artifacts.filtered_metadata_rows}")
    print(f"Generated images: {artifacts.total_generated_images}")
    print(f"Balanced train images: {artifacts.balanced_train_images}")
    print(f"Train original samples: {artifacts.train_original_samples}")
    print(f"Test original samples: {artifacts.test_original_samples}")
    print(f"Train samples: {artifacts.train_samples}")
    print(f"Test samples: {artifacts.test_samples}")
    print(f"Split strategy: {artifacts.split_strategy}")
    print(f"Split group columns: {', '.join(artifacts.split_group_columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
