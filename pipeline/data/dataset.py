from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import pydicom
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)
from sklearn.utils import resample

from pipeline.data.constants import (
    DEFAULT_AUGMENTATIONS_PER_IMAGE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SAMPLES_PER_CLASS,
    DEFAULT_TEST_SIZE,
    VALID_LABELS,
)


@dataclass(frozen=True)
class DatasetPreparationConfig:
    raw_data_dir: Path
    images_output_dir: Path
    splits_output_dir: Path
    metadata_filename: str = "INbreast.csv"
    dicom_subdir: str = "AllDICOMs"
    resize_dim: tuple[int, int] = DEFAULT_IMAGE_SIZE
    augmentations_per_image: int = DEFAULT_AUGMENTATIONS_PER_IMAGE
    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE

    @property
    def metadata_path(self) -> Path:
        return self.raw_data_dir / self.metadata_filename

    @property
    def dicom_dir(self) -> Path:
        return self.raw_data_dir / self.dicom_subdir

    @property
    def train_split_path(self) -> Path:
        return self.splits_output_dir / "train_split.csv"

    @property
    def test_split_path(self) -> Path:
        return self.splits_output_dir / "test_split.csv"

    @property
    def split_manifest_path(self) -> Path:
        return self.splits_output_dir / "split_summary.json"


@dataclass(frozen=True)
class DatasetPreparationArtifacts:
    train_split_path: Path
    test_split_path: Path
    images_output_dir: Path
    split_manifest_path: Path
    filtered_metadata_rows: int
    total_generated_images: int
    balanced_train_images: int
    train_original_samples: int
    test_original_samples: int
    train_samples: int
    test_samples: int
    split_strategy: str
    split_group_columns: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    strategy: str
    group_columns: tuple[str, ...]


def normalize_file_number(value: object) -> str:
    normalized = str(value).strip()
    if normalized.endswith(".0"):
        normalized = normalized[:-2]
    if normalized.isdigit():
        return str(int(normalized))
    return normalized


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(metadata_path, delimiter=";")
    dataframe.columns = dataframe.columns.str.strip()
    dataframe["File Name"] = dataframe["File Name"].map(normalize_file_number)
    dataframe["Bi-Rads"] = dataframe["Bi-Rads"].astype(str).str.strip().str.lower()
    dataframe = dataframe[dataframe["Bi-Rads"].isin(VALID_LABELS)].reset_index(
        drop=True
    )
    return dataframe


def _normalize_column_name(column_name: str) -> str:
    return "".join(
        character for character in column_name.lower() if character.isalnum()
    )


def _resolve_metadata_column(metadata: pd.DataFrame, *aliases: str) -> str | None:
    normalized_lookup = {
        _normalize_column_name(column): column for column in metadata.columns
    }
    for alias in aliases:
        resolved = normalized_lookup.get(_normalize_column_name(alias))
        if resolved is not None:
            return resolved
    return None


def _normalize_group_value(value: object) -> str:
    if pd.isna(value):
        return "missing"
    normalized = str(value).strip()
    return normalized if normalized else "missing"


def annotate_split_groups(
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, str, tuple[str, ...]]:
    annotated = metadata.copy()
    annotated["source_id"] = annotated["File Name"].map(normalize_file_number)

    patient_column = _resolve_metadata_column(
        annotated,
        "Patient ID",
        "Patient_ID",
        "Patient Number",
        "Patient",
    )
    exam_columns = tuple(
        column
        for column in (
            _resolve_metadata_column(
                annotated, "Accession Number", "Exam ID", "Exam_ID", "Study ID"
            ),
            _resolve_metadata_column(
                annotated, "Acquisition Date", "Study Date", "Exam Date"
            ),
            _resolve_metadata_column(annotated, "Laterality", "Side"),
            _resolve_metadata_column(annotated, "View", "View Position"),
        )
        if column is not None
    )

    if patient_column is not None:
        annotated["patient_id"] = annotated[patient_column].map(_normalize_group_value)
        annotated["split_group_id"] = annotated["patient_id"]
        return annotated, "patient", (patient_column,)

    if exam_columns:
        annotated["split_group_id"] = annotated.apply(
            lambda row: "|".join(
                _normalize_group_value(row[column]) for column in exam_columns
            ),
            axis=1,
        )
        return annotated, "exam", exam_columns

    annotated["split_group_id"] = annotated["source_id"]
    return annotated, "image", ("File Name",)


def build_dicom_mapping(dicom_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for file_name in dicom_dir.iterdir():
        if file_name.suffix.lower() != ".dcm":
            continue
        mapping[normalize_file_number(file_name.name.split("_")[0])] = file_name.name
    return mapping


def create_augmentation_pipeline() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=1.0,
            ),
            A.Rotate(limit=360, p=1.0, border_mode=cv2.BORDER_REPLICATE),
            A.Affine(scale=(0.6, 1.4), rotate=(-90, 90), p=1.0),
        ]
    )


def normalize_and_resize_dicom(dicom_path: Path, resize_dim: tuple[int, int]) -> tuple:
    dataset = pydicom.dcmread(dicom_path)
    image = dataset.pixel_array.astype("float32")
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
    resized = cv2.resize(image, resize_dim, interpolation=cv2.INTER_AREA)
    if getattr(resized, "ndim", 2) == 3 and resized.shape[2] == 1:
        resized = resized.squeeze(-1)
    return resized, dataset


def export_image_dataset(
    metadata: pd.DataFrame,
    dicom_mapping: dict[str, str],
    config: DatasetPreparationConfig,
    *,
    split_name: str,
    include_augmentations: bool,
) -> pd.DataFrame:
    split_output_dir = config.images_output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    augment = create_augmentation_pipeline() if include_augmentations else None

    generated_rows: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        file_number = row["source_id"]
        label = row["Bi-Rads"]
        dicom_name = dicom_mapping.get(file_number)
        if not dicom_name:
            continue

        dicom_path = config.dicom_dir / dicom_name
        image, _ = normalize_and_resize_dicom(dicom_path, config.resize_dim)
        image_uint8 = (image * 255).astype("uint8")

        original_path = split_output_dir / f"{file_number}_orig.png"
        cv2.imwrite(str(original_path), image_uint8)

        base_row: dict[str, object] = {
            "image_path": str(original_path),
            "label": label,
            "source_id": file_number,
            "split_group_id": row["split_group_id"],
            "split": split_name,
            "is_augmented": False,
            "augmentation_index": -1,
        }
        if "patient_id" in row.index:
            base_row["patient_id"] = row["patient_id"]
        generated_rows.append(base_row)

        if not include_augmentations or augment is None:
            continue

        for index in range(config.augmentations_per_image):
            augmented = augment(image=image)["image"]
            if getattr(augmented, "ndim", 2) == 3 and augmented.shape[2] == 1:
                augmented = augmented.squeeze(-1)
            augmented_uint8 = (augmented * 255).astype("uint8")
            augmented_path = split_output_dir / f"{file_number}_aug{index}.png"
            cv2.imwrite(str(augmented_path), augmented_uint8)
            generated_rows.append(
                {
                    **base_row,
                    "image_path": str(augmented_path),
                    "is_augmented": True,
                    "augmentation_index": index,
                }
            )

    return pd.DataFrame(generated_rows)


def balance_df_trim_above(
    dataframe: pd.DataFrame, samples_per_class: int, random_state: int
) -> pd.DataFrame:
    if dataframe.empty:
        raise ValueError(
            "No images were generated during dataset export. "
            "Check metadata labels and DICOM filename mapping."
        )

    balanced_frames: list[pd.DataFrame] = []
    for label in dataframe["label"].unique():
        subset = dataframe[dataframe["label"] == label]
        if len(subset) <= samples_per_class:
            balanced_frames.append(subset)
            continue
        trimmed = resample(
            subset,
            replace=False,
            n_samples=samples_per_class,
            random_state=random_state,
        )
        balanced_frames.append(trimmed)
    return pd.concat(balanced_frames, ignore_index=True)


def _estimate_split_count(test_size: float, total_groups: int) -> int:
    estimated = int(round(1.0 / test_size)) if test_size > 0 else 2
    return max(2, min(total_groups, estimated))


def _split_with_groups(
    dataframe: pd.DataFrame,
    *,
    label_column: str,
    group_column: str,
    test_size: float,
    random_state: int,
) -> DatasetSplitResult | None:
    groups = dataframe[group_column].astype(str)
    if groups.nunique() < 2:
        return None

    labels = dataframe[label_column]
    expected_test_samples = len(dataframe) * test_size
    n_splits = _estimate_split_count(test_size, groups.nunique())

    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        selected_split: tuple[list[int], list[int]] | None = None
        selected_delta: float | None = None
        for train_idx, test_idx in splitter.split(dataframe, labels, groups):
            current_delta = abs(len(test_idx) - expected_test_samples)
            if selected_delta is None or current_delta < selected_delta:
                selected_split = (train_idx.tolist(), test_idx.tolist())
                selected_delta = current_delta

        if selected_split is not None:
            train_idx, test_idx = selected_split
            return DatasetSplitResult(
                train_df=dataframe.iloc[train_idx].reset_index(drop=True),
                test_df=dataframe.iloc[test_idx].reset_index(drop=True),
                strategy="stratified_group",
                group_columns=(group_column,),
            )
    except ValueError:
        pass

    try:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(dataframe, labels, groups))
        return DatasetSplitResult(
            train_df=dataframe.iloc[train_idx].reset_index(drop=True),
            test_df=dataframe.iloc[test_idx].reset_index(drop=True),
            strategy="group_shuffle",
            group_columns=(group_column,),
        )
    except ValueError:
        return None


def split_dataset(
    dataframe: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    label_column: str = "label",
    group_column: str | None = None,
) -> DatasetSplitResult:
    resolved_group_column = group_column if group_column in dataframe.columns else None
    if resolved_group_column is None and "split_group_id" in dataframe.columns:
        resolved_group_column = "split_group_id"

    if resolved_group_column is not None:
        grouped_split = _split_with_groups(
            dataframe,
            label_column=label_column,
            group_column=resolved_group_column,
            test_size=test_size,
            random_state=random_state,
        )
        if grouped_split is not None:
            return grouped_split

    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=dataframe[label_column],
    )
    return DatasetSplitResult(
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        strategy="image_stratified",
        group_columns=(
            tuple() if resolved_group_column is None else (resolved_group_column,)
        ),
    )


def _class_distribution(dataframe: pd.DataFrame, label_column: str) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in dataframe[label_column].value_counts().sort_index().items()
    }


def _build_split_manifest(
    *,
    config: DatasetPreparationConfig,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_strategy: str,
    split_group_columns: tuple[str, ...],
) -> dict[str, object]:
    train_source_ids = set(train_metadata["source_id"].astype(str))
    test_source_ids = set(test_metadata["source_id"].astype(str))
    train_group_ids = set(train_metadata["split_group_id"].astype(str))
    test_group_ids = set(test_metadata["split_group_id"].astype(str))
    return {
        "random_state": config.random_state,
        "test_size": config.test_size,
        "split_strategy": split_strategy,
        "split_group_columns": list(split_group_columns),
        "augmentations_per_image": config.augmentations_per_image,
        "samples_per_class": config.samples_per_class,
        "filtered_metadata_rows": int(len(train_metadata) + len(test_metadata)),
        "train_original_samples": int(len(train_metadata)),
        "test_original_samples": int(len(test_metadata)),
        "train_images_after_balancing": int(len(train_df)),
        "test_images": int(len(test_df)),
        "train_label_distribution_original": _class_distribution(
            train_metadata, "Bi-Rads"
        ),
        "test_label_distribution_original": _class_distribution(
            test_metadata, "Bi-Rads"
        ),
        "train_label_distribution_images": _class_distribution(train_df, "label"),
        "test_label_distribution_images": _class_distribution(test_df, "label"),
        "shared_source_ids": int(len(train_source_ids & test_source_ids)),
        "shared_split_group_ids": int(len(train_group_ids & test_group_ids)),
    }


def prepare_dataset(config: DatasetPreparationConfig) -> DatasetPreparationArtifacts:
    config.images_output_dir.mkdir(parents=True, exist_ok=True)
    config.splits_output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(config.metadata_path)
    metadata, split_group_level, split_group_columns = annotate_split_groups(metadata)
    dicom_mapping = build_dicom_mapping(config.dicom_dir)
    metadata_split = split_dataset(
        metadata,
        test_size=config.test_size,
        random_state=config.random_state,
        label_column="Bi-Rads",
        group_column="split_group_id",
    )

    train_generated_df = export_image_dataset(
        metadata_split.train_df,
        dicom_mapping,
        config,
        split_name="train",
        include_augmentations=True,
    )
    test_generated_df = export_image_dataset(
        metadata_split.test_df,
        dicom_mapping,
        config,
        split_name="test",
        include_augmentations=False,
    )
    balanced_train_df = balance_df_trim_above(
        train_generated_df,
        samples_per_class=config.samples_per_class,
        random_state=config.random_state,
    )

    balanced_train_df.to_csv(config.train_split_path, index=False)
    test_generated_df.to_csv(config.test_split_path, index=False)

    split_strategy = (
        f"grouped_{split_group_level}"
        if split_group_level != "image"
        else metadata_split.strategy
    )
    split_manifest = _build_split_manifest(
        config=config,
        train_metadata=metadata_split.train_df,
        test_metadata=metadata_split.test_df,
        train_df=balanced_train_df,
        test_df=test_generated_df,
        split_strategy=split_strategy,
        split_group_columns=split_group_columns,
    )
    config.split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    return DatasetPreparationArtifacts(
        train_split_path=config.train_split_path,
        test_split_path=config.test_split_path,
        images_output_dir=config.images_output_dir,
        split_manifest_path=config.split_manifest_path,
        filtered_metadata_rows=len(metadata),
        total_generated_images=len(train_generated_df) + len(test_generated_df),
        balanced_train_images=len(balanced_train_df),
        train_original_samples=len(metadata_split.train_df),
        test_original_samples=len(metadata_split.test_df),
        train_samples=len(balanced_train_df),
        test_samples=len(test_generated_df),
        split_strategy=split_strategy,
        split_group_columns=split_group_columns,
    )
