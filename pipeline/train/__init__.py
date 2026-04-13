"""Training components for the reusable project pipeline."""

__all__ = [
    "LEGACY_PREPROCESSING_METHODS",
    "MODEL_BUILDERS",
    "ModelRuntimeConfig",
    "PreprocessingTask",
    "SINGLE_PREPROCESSING_METHODS",
    "TrainingConfig",
    "TrainingTask",
    "build_training_tasks",
    "iter_preprocessing_tasks",
    "run_training_task",
    "run_training_pipeline",
    "save_run_result",
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
    if name in {
        "TrainingConfig",
        "TrainingTask",
        "build_training_tasks",
        "run_training_task",
        "run_training_pipeline",
        "save_run_result",
    }:
        from pipeline.train.runner import (
            TrainingConfig,
            TrainingTask,
            build_training_tasks,
            run_training_pipeline,
            run_training_task,
            save_run_result,
        )

        exports = {
            "TrainingConfig": TrainingConfig,
            "TrainingTask": TrainingTask,
            "build_training_tasks": build_training_tasks,
            "run_training_task": run_training_task,
            "run_training_pipeline": run_training_pipeline,
            "save_run_result": save_run_result,
        }
        return exports[name]
    raise AttributeError(f"module 'pipeline.train' has no attribute {name!r}")
