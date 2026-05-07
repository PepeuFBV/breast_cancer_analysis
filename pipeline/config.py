from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.data.constants import DEFAULT_RANDOM_STATE
from pipeline.train.models import MODEL_BUILDERS, ModelRuntimeConfig
from pipeline.utils.paths import PROJECT_ROOT, ProjectPaths, build_project_paths
from pipeline.utils.runtime import resolve_bool_flag
from pipeline.utils.runtime_limits import CpuExecutionLimits, validate_cpu_execution_limits

if TYPE_CHECKING:
    from pipeline.data.dataset import DatasetPreparationConfig
    from pipeline.evaluate.reporting import EvaluationConfig
    from pipeline.train.runner import TrainingConfig


DEFAULT_EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.default.json"
AVAILABLE_PREPROCESSING_IDS = (
    "none",
    "denoise",
    "binarize",
    "lowpass",
    "erode",
    "dilate",
    "open",
    "close",
    "clahe",
)


def _resolve_project_relative_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _normalize_grid_values(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalize_grid_values(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_grid_values(item) for key, item in value.items()}
    return value


def _normalize_image_size(values: list[int] | tuple[int, int]) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError(f"image_size must contain exactly two values, got {values!r}")
    width, height = values
    return int(width), int(height)


def _normalize_augmentation_values(value: Any) -> tuple[int, ...]:
    if isinstance(value, list):
        if not value:
            raise ValueError("preprocess.augmentations_per_image list cannot be empty.")
        normalized = tuple(int(item) for item in value)
    else:
        normalized = (int(value),)
    invalid = [item for item in normalized if item < 0]
    if invalid:
        raise ValueError("preprocess.augmentations_per_image values must be >= 0, " f"got {invalid}.")
    return normalized


def _validate_model_names(model_names: list[str] | None, configured_models: dict[str, ModelRuntimeConfig]) -> None:
    available_models = set(MODEL_BUILDERS)
    requested_models = set(model_names or [])
    unknown_requested = sorted(requested_models - available_models)
    if unknown_requested:
        raise ValueError(f"Unknown model names in train.model_names: {unknown_requested}")

    unknown_configured = sorted(set(configured_models) - available_models)
    if unknown_configured:
        raise ValueError(f"Unknown model names in models config: {unknown_configured}")


def _is_valid_preprocessing_selection(preproc_id: str) -> bool:
    available = set(AVAILABLE_PREPROCESSING_IDS)
    if preproc_id in available:
        return True
    if "__" not in preproc_id:
        return False
    parts = preproc_id.split("__")
    if len(parts) != 2:
        return False
    return all(part in available and part != "none" for part in parts)


def _validate_preprocessing_config(
    preprocessing_ids: list[str] | None,
    preprocessing_grid: dict[str, dict[str, list[Any]]],
) -> None:
    available = set(AVAILABLE_PREPROCESSING_IDS)
    unknown_grid = sorted(set(preprocessing_grid) - available)
    if unknown_grid:
        raise ValueError("Unknown preprocessing ids in preprocess.preprocessing_grid: " f"{unknown_grid}")

    for preproc_id, param_space in preprocessing_grid.items():
        if not isinstance(param_space, dict):
            raise ValueError(f"Parameter grid for preprocessing '{preproc_id}' " "must be a JSON object.")
        for param_name, values in param_space.items():
            if not isinstance(values, list):
                raise ValueError(f"Parameter '{param_name}' for preprocessing " f"'{preproc_id}' must be a JSON array.")

    invalid_requested = sorted(preproc_id for preproc_id in (preprocessing_ids or []) if not _is_valid_preprocessing_selection(preproc_id))
    if invalid_requested:
        raise ValueError(f"Unknown preprocessing ids in train.preprocessing_ids: {invalid_requested}")


@dataclass(frozen=True)
class ExperimentPathsConfig:
    raw_data_dir: Path
    artifacts_dir: Path
    train_split: Path | None = None
    test_split: Path | None = None
    history_dir: Path | None = None
    predictions_dir: Path | None = None
    report_output: Path | None = None


@dataclass(frozen=True)
class ExperimentPreprocessConfig:
    image_size: tuple[int, int]
    augmentations_per_image: tuple[int, ...]
    samples_per_class: int
    test_size: float
    random_state: int
    preprocessing_grid: dict[str, dict[str, list[Any]]]


@dataclass(frozen=True)
class ExperimentTrainConfig:
    model_names: list[str] | None
    preprocessing_ids: list[str] | None
    include_combinations: bool
    folds: int
    validation_size: float
    batch_size: int
    epochs: int
    learning_rate: float
    loss: str
    run_skip: bool
    random_state: int


@dataclass(frozen=True)
class ExperimentEvaluateConfig:
    top_k: int


@dataclass(frozen=True)
class ExperimentRunnerConfig:
    isolate_tasks: bool
    task_cooldown_seconds: float
    task_timeout_seconds: float | None
    device_policy: str
    gpu_retries: int
    cpu_retries: int
    cooldown_after_oom_seconds: float
    gpu_recovery_cooldown_seconds: float
    max_consecutive_oom: int
    max_task_attempts: int
    fail_fast_on_oom: bool
    cpu_max_threads: int | None = None
    cpu_opencv_threads: int | None = None
    cpu_inter_op_threads: int | None = None
    cpu_intra_op_threads: int | None = None
    cpu_nice: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    paths: ExperimentPathsConfig
    preprocess: ExperimentPreprocessConfig
    train: ExperimentTrainConfig
    runner: ExperimentRunnerConfig
    models: dict[str, ModelRuntimeConfig]
    evaluate: ExperimentEvaluateConfig

    def resolve_project_paths(
        self,
        *,
        raw_data_dir: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> ProjectPaths:
        resolved_raw_data_dir = raw_data_dir if raw_data_dir is not None else self.paths.raw_data_dir
        resolved_artifacts_dir = artifacts_dir if artifacts_dir is not None else self.paths.artifacts_dir
        return build_project_paths(resolved_raw_data_dir, resolved_artifacts_dir)

    def build_dataset_preparation_config(
        self,
        project_paths: ProjectPaths,
        *,
        image_size: tuple[int, int] | None = None,
        augmentations_per_image: int | None = None,
        samples_per_class: int | None = None,
        test_size: float | None = None,
        random_state: int | None = None,
    ) -> DatasetPreparationConfig:
        from pipeline.data.dataset import DatasetPreparationConfig

        default_augmentation = self.preprocess.augmentations_per_image[0]
        return DatasetPreparationConfig(
            raw_data_dir=project_paths.raw_data_dir,
            images_output_dir=project_paths.processed_images_dir,
            splits_output_dir=project_paths.processed_splits_dir,
            resize_dim=image_size or self.preprocess.image_size,
            augmentations_per_image=(default_augmentation if augmentations_per_image is None else augmentations_per_image),
            samples_per_class=(self.preprocess.samples_per_class if samples_per_class is None else samples_per_class),
            test_size=self.preprocess.test_size if test_size is None else test_size,
            random_state=(self.preprocess.random_state if random_state is None else random_state),
        )

    def build_training_config(
        self,
        project_paths: ProjectPaths,
        *,
        train_split: str | Path | None = None,
        test_split: str | Path | None = None,
        history_dir: str | Path | None = None,
        predictions_dir: str | Path | None = None,
        folds: int | None = None,
        validation_size: float | None = None,
        batch_size: int | None = None,
        epochs: int | None = None,
        learning_rate: float | None = None,
        loss: str | None = None,
        random_state: int | None = None,
        model_names: list[str] | None = None,
        preprocessing_ids: list[str] | None = None,
        include_combinations: bool | None = None,
        augmentations_per_image: int | list[int] | None = None,
        run_skip: bool | None = None,
    ) -> TrainingConfig:
        from pipeline.train.runner import TrainingConfig

        if augmentations_per_image is None:
            augmentation_values = self.preprocess.augmentations_per_image
        else:
            augmentation_values = _normalize_augmentation_values(augmentations_per_image)

        return TrainingConfig(
            train_split_path=_resolve_configured_path(train_split, self.paths.train_split, project_paths.train_split_path),
            test_split_path=_resolve_configured_path(test_split, self.paths.test_split, project_paths.test_split_path),
            history_dir=_resolve_configured_path(history_dir, self.paths.history_dir, project_paths.history_dir),
            predictions_dir=_resolve_configured_path(
                predictions_dir,
                self.paths.predictions_dir,
                project_paths.predictions_dir,
            ),
            folds=self.train.folds if folds is None else folds,
            validation_size=(self.train.validation_size if validation_size is None else validation_size),
            batch_size=self.train.batch_size if batch_size is None else batch_size,
            epochs=self.train.epochs if epochs is None else epochs,
            run_skip=resolve_bool_flag(run_skip, default=self.train.run_skip),
            model_names=self.train.model_names if model_names is None else model_names,
            preprocessing_ids=(self.train.preprocessing_ids if preprocessing_ids is None else preprocessing_ids),
            include_combinations=resolve_bool_flag(
                include_combinations,
                default=self.train.include_combinations,
            ),
            loss=self.train.loss if loss is None else loss,
            learning_rate=(self.train.learning_rate if learning_rate is None else learning_rate),
            random_state=(self.train.random_state if random_state is None else random_state),
            model_runtime=self.models,
            preprocessing_grids=self.preprocess.preprocessing_grid,
            augmentation_values=augmentation_values,
        )

    def build_cpu_execution_limits(
        self,
        *,
        cpu_max_threads: int | None = None,
        cpu_opencv_threads: int | None = None,
        cpu_inter_op_threads: int | None = None,
        cpu_intra_op_threads: int | None = None,
        cpu_nice: int | None = None,
    ) -> CpuExecutionLimits:
        return validate_cpu_execution_limits(
            CpuExecutionLimits(
                max_threads=(self.runner.cpu_max_threads if cpu_max_threads is None else cpu_max_threads),
                opencv_threads=(self.runner.cpu_opencv_threads if cpu_opencv_threads is None else cpu_opencv_threads),
                inter_op_threads=(self.runner.cpu_inter_op_threads if cpu_inter_op_threads is None else cpu_inter_op_threads),
                intra_op_threads=(self.runner.cpu_intra_op_threads if cpu_intra_op_threads is None else cpu_intra_op_threads),
                nice=(self.runner.cpu_nice if cpu_nice is None else cpu_nice),
            )
        )

    def build_evaluation_config(
        self,
        project_paths: ProjectPaths,
        *,
        history_dir: str | Path | None = None,
        predictions_dir: str | Path | None = None,
        output_path: str | Path | None = None,
        details_dir: str | Path | None = None,
        top_k: int | None = None,
    ) -> EvaluationConfig:
        from pipeline.evaluate.reporting import EvaluationConfig

        return EvaluationConfig(
            history_dir=_resolve_configured_path(history_dir, self.paths.history_dir, project_paths.history_dir),
            predictions_dir=_resolve_configured_path(
                predictions_dir,
                self.paths.predictions_dir,
                project_paths.predictions_dir,
            ),
            output_path=_resolve_configured_path(
                output_path,
                self.paths.report_output,
                project_paths.final_report_path,
            ),
            details_dir=(None if details_dir is None else _resolve_project_relative_path(details_dir)),
            top_k=self.evaluate.top_k if top_k is None else top_k,
        )


def _resolve_configured_path(
    explicit_value: str | Path | None,
    config_value: Path | None,
    default_value: Path,
) -> Path:
    if explicit_value is not None:
        explicit_path = Path(explicit_value)
        return explicit_path if explicit_path.is_absolute() else PROJECT_ROOT / explicit_path
    if config_value is not None:
        return config_value
    return default_value


def _model_runtime_from_dict(config: dict[str, Any]) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        input_channels=int(config.get("input_channels", 3)),
        batch_size=(None if config.get("batch_size") is None else int(config["batch_size"])),
        dense_units=(None if config.get("dense_units") is None else int(config["dense_units"])),
        dropout_rate=(None if config.get("dropout_rate") is None else float(config["dropout_rate"])),
        dropout_rates=tuple(float(value) for value in config.get("dropout_rates", [])),
    )


def _optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be > 0 when provided.")
    return resolved


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    config_path = _resolve_project_relative_path(path) or DEFAULT_EXPERIMENT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)

    raw_preprocessing_grid = raw_config["preprocess"]["preprocessing_grid"]
    _validate_preprocessing_config(
        raw_config["train"].get("preprocessing_ids"),
        raw_preprocessing_grid,
    )
    preprocessing_grid = {preproc_id: _normalize_grid_values(param_space) for preproc_id, param_space in raw_preprocessing_grid.items()}
    model_config = {model_name: _model_runtime_from_dict(model_values) for model_name, model_values in raw_config["models"].items()}
    raw_runner = dict(raw_config.get("runner", {}))

    _validate_model_names(raw_config["train"].get("model_names"), model_config)
    cooldown_seconds = float(raw_runner.get("task_cooldown_seconds", 0.0))
    if cooldown_seconds < 0:
        raise ValueError("runner.task_cooldown_seconds must be >= 0.")
    timeout_raw = raw_runner.get("task_timeout_seconds")
    timeout_seconds = None if timeout_raw is None else float(timeout_raw)
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("runner.task_timeout_seconds must be > 0 when provided.")
    device_policy = str(raw_runner.get("device_policy", "adaptive"))
    if device_policy not in {"gpu-first", "cpu-only", "gpu-only", "adaptive"}:
        raise ValueError("runner.device_policy must be one of: gpu-first, cpu-only, gpu-only, adaptive.")
    gpu_retries = int(raw_runner.get("gpu_retries", 1))
    cpu_retries = int(raw_runner.get("cpu_retries", 1))
    if gpu_retries < 0:
        raise ValueError("runner.gpu_retries must be >= 0.")
    if cpu_retries < 0:
        raise ValueError("runner.cpu_retries must be >= 0.")
    cooldown_after_oom_seconds = float(raw_runner.get("cooldown_after_oom_seconds", 15.0))
    if cooldown_after_oom_seconds < 0:
        raise ValueError("runner.cooldown_after_oom_seconds must be >= 0.")
    gpu_recovery_cooldown_seconds = float(raw_runner.get("gpu_recovery_cooldown_seconds", 60.0))
    if gpu_recovery_cooldown_seconds < 0:
        raise ValueError("runner.gpu_recovery_cooldown_seconds must be >= 0.")
    max_consecutive_oom = int(raw_runner.get("max_consecutive_oom", 3))
    if max_consecutive_oom <= 0:
        raise ValueError("runner.max_consecutive_oom must be > 0.")
    max_task_attempts = int(raw_runner.get("max_task_attempts", 4))
    if max_task_attempts <= 0:
        raise ValueError("runner.max_task_attempts must be > 0.")
    fail_fast_on_oom = bool(raw_runner.get("fail_fast_on_oom", False))
    cpu_limits = validate_cpu_execution_limits(
        CpuExecutionLimits(
            max_threads=_optional_positive_int("runner.cpu_max_threads", raw_runner.get("cpu_max_threads")),
            opencv_threads=_optional_positive_int("runner.cpu_opencv_threads", raw_runner.get("cpu_opencv_threads")),
            inter_op_threads=_optional_positive_int("runner.cpu_inter_op_threads", raw_runner.get("cpu_inter_op_threads")),
            intra_op_threads=_optional_positive_int("runner.cpu_intra_op_threads", raw_runner.get("cpu_intra_op_threads")),
            nice=(None if raw_runner.get("cpu_nice") is None else int(raw_runner.get("cpu_nice"))),
        )
    )

    return ExperimentConfig(
        source_path=config_path,
        paths=ExperimentPathsConfig(
            raw_data_dir=_resolve_project_relative_path(raw_config["paths"]["raw_data_dir"]) or PROJECT_ROOT,
            artifacts_dir=_resolve_project_relative_path(raw_config["paths"]["artifacts_dir"]) or PROJECT_ROOT,
            train_split=_resolve_project_relative_path(raw_config["paths"].get("train_split")),
            test_split=_resolve_project_relative_path(raw_config["paths"].get("test_split")),
            history_dir=_resolve_project_relative_path(raw_config["paths"].get("history_dir")),
            predictions_dir=_resolve_project_relative_path(raw_config["paths"].get("predictions_dir")),
            report_output=_resolve_project_relative_path(raw_config["paths"].get("report_output")),
        ),
        preprocess=ExperimentPreprocessConfig(
            image_size=_normalize_image_size(raw_config["preprocess"]["image_size"]),
            augmentations_per_image=_normalize_augmentation_values(raw_config["preprocess"]["augmentations_per_image"]),
            samples_per_class=int(raw_config["preprocess"]["samples_per_class"]),
            test_size=float(raw_config["preprocess"]["test_size"]),
            random_state=int(raw_config["preprocess"].get("random_state", DEFAULT_RANDOM_STATE)),
            preprocessing_grid=preprocessing_grid,
        ),
        train=ExperimentTrainConfig(
            model_names=raw_config["train"].get("model_names"),
            preprocessing_ids=raw_config["train"].get("preprocessing_ids"),
            include_combinations=bool(raw_config["train"]["include_combinations"]),
            folds=int(raw_config["train"]["folds"]),
            validation_size=float(raw_config["train"].get("validation_size", 0.2)),
            batch_size=int(raw_config["train"]["batch_size"]),
            epochs=int(raw_config["train"]["epochs"]),
            learning_rate=float(raw_config["train"]["learning_rate"]),
            loss=str(raw_config["train"]["loss"]),
            run_skip=bool(raw_config["train"]["run_skip"]),
            random_state=int(raw_config["train"].get("random_state", DEFAULT_RANDOM_STATE)),
        ),
        runner=ExperimentRunnerConfig(
            isolate_tasks=bool(raw_runner.get("isolate_tasks", False)),
            task_cooldown_seconds=cooldown_seconds,
            task_timeout_seconds=timeout_seconds,
            device_policy=device_policy,
            gpu_retries=gpu_retries,
            cpu_retries=cpu_retries,
            cooldown_after_oom_seconds=cooldown_after_oom_seconds,
            gpu_recovery_cooldown_seconds=gpu_recovery_cooldown_seconds,
            max_consecutive_oom=max_consecutive_oom,
            max_task_attempts=max_task_attempts,
            fail_fast_on_oom=fail_fast_on_oom,
            cpu_max_threads=cpu_limits.max_threads,
            cpu_opencv_threads=cpu_limits.opencv_threads,
            cpu_inter_op_threads=cpu_limits.inter_op_threads,
            cpu_intra_op_threads=cpu_limits.intra_op_threads,
            cpu_nice=cpu_limits.nice,
        ),
        models=model_config,
        evaluate=ExperimentEvaluateConfig(
            top_k=int(raw_config["evaluate"]["top_k"]),
        ),
    )
