"""Dataset preparation helpers for the project pipeline."""

from pipeline.data.constants import DEFAULT_IMAGE_SIZE, LABEL_MAPPING, VALID_LABELS

__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DatasetPreparationArtifacts",
    "DatasetPreparationConfig",
    "LABEL_MAPPING",
    "VALID_LABELS",
    "prepare_dataset",
]


def __getattr__(name: str):
    if name in {"DatasetPreparationArtifacts", "DatasetPreparationConfig", "prepare_dataset"}:
        from pipeline.data.dataset import DatasetPreparationArtifacts, DatasetPreparationConfig, prepare_dataset

        exports = {
            "DatasetPreparationArtifacts": DatasetPreparationArtifacts,
            "DatasetPreparationConfig": DatasetPreparationConfig,
            "prepare_dataset": prepare_dataset,
        }
        return exports[name]
    raise AttributeError(f"module 'pipeline.data' has no attribute {name!r}")
