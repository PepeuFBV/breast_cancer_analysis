"""Training components for the reusable project pipeline."""

__all__ = [
    "LEGACY_PREPROCESSING_METHODS",
    "MODEL_BUILDERS",
    "ModelRuntimeConfig",
    "PreprocessingTask",
    "SINGLE_PREPROCESSING_METHODS",
    "TrainingConfig",
    "iter_preprocessing_tasks",
    "run_training_pipeline",
]


def __getattr__(name: str):
    if name in {"MODEL_BUILDERS", "ModelRuntimeConfig"}:
        from pipeline.train.models import MODEL_BUILDERS, ModelRuntimeConfig

        exports = {
            "MODEL_BUILDERS": MODEL_BUILDERS,
            "ModelRuntimeConfig": ModelRuntimeConfig,
        }
        return exports[name]
    if name in {
        "LEGACY_PREPROCESSING_METHODS",
        "SINGLE_PREPROCESSING_METHODS",
        "PreprocessingTask",
        "iter_preprocessing_tasks",
    }:
        from pipeline.train.preprocessing import (
            LEGACY_PREPROCESSING_METHODS,
            SINGLE_PREPROCESSING_METHODS,
            PreprocessingTask,
            iter_preprocessing_tasks,
        )

        exports = {
            "LEGACY_PREPROCESSING_METHODS": LEGACY_PREPROCESSING_METHODS,
            "SINGLE_PREPROCESSING_METHODS": SINGLE_PREPROCESSING_METHODS,
            "PreprocessingTask": PreprocessingTask,
            "iter_preprocessing_tasks": iter_preprocessing_tasks,
        }
        return exports[name]
    if name in {"TrainingConfig", "run_training_pipeline"}:
        from pipeline.train.runner import TrainingConfig, run_training_pipeline

        exports = {
            "TrainingConfig": TrainingConfig,
            "run_training_pipeline": run_training_pipeline,
        }
        return exports[name]
    raise AttributeError(f"module 'pipeline.train' has no attribute {name!r}")
