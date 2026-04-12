from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
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


@dataclass(frozen=True)
class DatasetPreparationArtifacts:
    train_split_path: Path
    test_split_path: Path
    images_output_dir: Path
    filtered_metadata_rows: int
    total_generated_images: int
    balanced_images: int
    train_samples: int
    test_samples: int


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
    dataframe = dataframe[dataframe["Bi-Rads"].isin(VALID_LABELS)].reset_index(drop=True)
    return dataframe


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


def export_augmented_dataset(
    metadata: pd.DataFrame,
    dicom_mapping: dict[str, str],
    config: DatasetPreparationConfig,
) -> pd.DataFrame:
    config.images_output_dir.mkdir(parents=True, exist_ok=True)
    augment = create_augmentation_pipeline()

    generated_rows: list[dict[str, str]] = []
    for _, row in metadata.iterrows():
        file_number = row["File Name"]
        label = row["Bi-Rads"]
        dicom_name = dicom_mapping.get(file_number)
        if not dicom_name:
            continue

        dicom_path = config.dicom_dir / dicom_name
        image, _ = normalize_and_resize_dicom(dicom_path, config.resize_dim)
        image_uint8 = (image * 255).astype("uint8")

        original_path = config.images_output_dir / f"{file_number}_orig.png"
        cv2.imwrite(str(original_path), image_uint8)
        generated_rows.append({"image_path": str(original_path), "label": label})

        for index in range(config.augmentations_per_image):
            augmented = augment(image=image)["image"]
            if getattr(augmented, "ndim", 2) == 3 and augmented.shape[2] == 1:
                augmented = augmented.squeeze(-1)
            augmented_uint8 = (augmented * 255).astype("uint8")
            augmented_path = config.images_output_dir / f"{file_number}_aug{index}.png"
            cv2.imwrite(str(augmented_path), augmented_uint8)
            generated_rows.append({"image_path": str(augmented_path), "label": label})

    return pd.DataFrame(generated_rows)


def balance_df_trim_above(dataframe: pd.DataFrame, samples_per_class: int, random_state: int) -> pd.DataFrame:
    if dataframe.empty:
        raise ValueError("No images were generated during dataset export. Check metadata labels and DICOM filename mapping.")

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


def split_dataset(dataframe: pd.DataFrame, *, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        dataframe[["image_path", "label"]],
        test_size=test_size,
        random_state=random_state,
        stratify=dataframe["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_dataset(config: DatasetPreparationConfig) -> DatasetPreparationArtifacts:
    config.images_output_dir.mkdir(parents=True, exist_ok=True)
    config.splits_output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(config.metadata_path)
    dicom_mapping = build_dicom_mapping(config.dicom_dir)
    augmented_df = export_augmented_dataset(metadata, dicom_mapping, config)
    balanced_df = balance_df_trim_above(
        augmented_df,
        samples_per_class=config.samples_per_class,
        random_state=config.random_state,
    )
    train_df, test_df = split_dataset(
        balanced_df,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    train_df.to_csv(config.train_split_path, index=False)
    test_df.to_csv(config.test_split_path, index=False)

    return DatasetPreparationArtifacts(
        train_split_path=config.train_split_path,
        test_split_path=config.test_split_path,
        images_output_dir=config.images_output_dir,
        filtered_metadata_rows=len(metadata),
        total_generated_images=len(augmented_df),
        balanced_images=len(balanced_df),
        train_samples=len(train_df),
        test_samples=len(test_df),
    )
