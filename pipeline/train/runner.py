from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from pipeline.data.constants import DEFAULT_RANDOM_STATE, LABEL_MAPPING
from pipeline.data.dataset import DatasetSplitResult, split_dataset
from pipeline.train.models import MODEL_BUILDERS, ModelBuilder, ModelRuntimeConfig
from pipeline.train.preprocessing import PreprocessingTask, iter_preprocessing_tasks
from pipeline.utils.gpu_env import configure_gpu_memory_growth
from pipeline.utils.memory import clear_ml_memory
from pipeline.utils.reproducibility import enforce_reproducibility
from pipeline.utils.runtime import format_duration

PhaseObserver = Callable[[str, dict[str, Any] | None], None]


def _emit_phase(
    phase_observer: PhaseObserver | None,
    phase: str,
    details: dict[str, Any] | None = None,
) -> None:
    if phase_observer is None:
        return
    try:
        phase_observer(phase, details)
    except Exception:
        # Observability callbacks must not affect training behavior.
        return


@dataclass(frozen=True)
class TrainingConfig:
    train_split_path: Path
    test_split_path: Path
    history_dir: Path
    predictions_dir: Path
    folds: int = 4
    validation_size: float = 0.2
    batch_size: int = 8
    epochs: int = 15
    num_classes: int = len(LABEL_MAPPING)
    run_skip: bool = True
    random_state: int = DEFAULT_RANDOM_STATE
    model_names: list[str] | None = None
    preprocessing_ids: list[str] | None = None
    include_combinations: bool = True
    loss: str = "categorical_crossentropy"
    learning_rate: float = 1e-4
    model_runtime: dict[str, ModelRuntimeConfig] | None = None
    preprocessing_grids: dict[str, dict[str, list[Any]]] | None = None
    augmentation_values: tuple[int, ...] = (1,)


@dataclass(frozen=True)
class TrainingTask:
    model_name: str
    preprocessing_task: PreprocessingTask
    augmentations_per_image: int = 1

    @property
    def preproc_id(self) -> str:
        return self.preprocessing_task.preproc_id

    @property
    def params(self) -> dict[str, Any]:
        return self.preprocessing_task.params

    @property
    def param_display(self) -> str:
        return self.preprocessing_task.param_display

    @property
    def param_id(self) -> str:
        return self.preprocessing_task.param_id

    @property
    def param_json(self) -> str:
        return self.preprocessing_task.param_json

    @property
    def is_combined(self) -> bool:
        return self.preprocessing_task.is_combined

    @property
    def label(self) -> str:
        return (
            f"{self.preproc_id} [{self.model_name} - {self.param_display} - "
            f"aug={self.augmentations_per_image}]"
        )


@dataclass(frozen=True)
class TrainingRunResult:
    preproc_id: str
    model_name: str
    param_id: str
    param_combo: str
    param_json: str
    best_val_acc: float
    best_epoch: int
    fold: int | None
    history_dict: dict[str, list[float]]
    predictions_df: pd.DataFrame
    selection_strategy: str
    train_samples: int
    validation_samples: int
    test_samples: int
    augmentations_per_image: int = 1
    summary_metrics: dict[str, Any] = field(default_factory=dict)


def load_split_dataframe(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    raw_labels = dataframe["label"].astype(str).str.strip().str.lower()
    mapped_labels = raw_labels.map(LABEL_MAPPING)
    if mapped_labels.isna().any():
        invalid_rows = raw_labels[mapped_labels.isna()]
        raise ValueError(f"Invalid labels found in split {path}: {sorted(invalid_rows.unique())}")
    dataframe["label"] = mapped_labels.astype(int)
    return dataframe


def preprocess_images(dataframe: pd.DataFrame, preproc_fn) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for _, row in dataframe.iterrows():
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {row['image_path']}")
        processed = preproc_fn(image)
        images.append(processed)
        labels.append(int(row["label"]))
    return np.stack(images), np.asarray(labels)


def _default_model_runtime(model_name: str, batch_size: int) -> ModelRuntimeConfig:
    if model_name.lower() in {"custom cnn", "bcnet"}:
        return ModelRuntimeConfig(input_channels=1, batch_size=batch_size)
    return ModelRuntimeConfig(input_channels=3, batch_size=4)


def _prepare_model_inputs(
    images: np.ndarray,
    model_name: str,
    batch_size: int,
    model_runtime: ModelRuntimeConfig | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    runtime = model_runtime or _default_model_runtime(model_name, batch_size)
    images = np.expand_dims(images, -1).astype("float32") / 255.0
    effective_batch_size = runtime.batch_size or batch_size
    if runtime.input_channels == 1:
        return images, tuple(images.shape[1:]), effective_batch_size

    if runtime.input_channels == 3:
        images = np.repeat(images, 3, axis=-1)
        return images, tuple(images.shape[1:]), effective_batch_size

    raise ValueError("Unsupported input_channels=" f"{runtime.input_channels} configured for model '{model_name}'.")


def _one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype="float32")[labels]


def _is_resource_exhausted_error(error: Exception) -> bool:
    error_name = error.__class__.__name__
    return error_name in ("ResourceExhaustedError", "InternalError") or "OOM" in str(error)


def _clear_keras_session() -> None:
    clear_ml_memory()


def _build_fit_callbacks() -> list[object]:
    try:
        from keras.callbacks import EarlyStopping

        return [
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=3,
                restore_best_weights=True,
            )
        ]
    except Exception:
        return []


def _prediction_trace_columns(dataframe: pd.DataFrame) -> list[str]:
    preferred = [
        "image_path",
        "source_id",
        "split_group_id",
        "patient_id",
        "split",
        "is_augmented",
        "augmentation_index",
    ]
    return [column for column in preferred if column in dataframe.columns]


def _predict_dataframe(
    model: object,
    model_inputs: np.ndarray,
    evaluation_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    probabilities = model.predict(model_inputs, batch_size=batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    prediction_df = pd.DataFrame(
        {
            "y_true": evaluation_df["label"].values,
            "y_pred": predictions,
            "y_pred_probability": probabilities.tolist(),
        }
    )
    for column in _prediction_trace_columns(evaluation_df):
        prediction_df[column] = evaluation_df[column].values
    for index in range(probabilities.shape[1]):
        prediction_df[f"prob_class_{index}"] = probabilities[:, index]
    probabilities = None
    predictions = None
    return prediction_df


def run_model_with_preprocessing(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    task: PreprocessingTask,
    model_name: str,
    model_fn: ModelBuilder,
    *,
    num_classes: int,
    batch_size: int,
    epochs: int,
    loss: str,
    learning_rate: float,
    random_state: int,
    model_runtime: ModelRuntimeConfig | None = None,
    max_retries: int = 3,
    phase_observer: PhaseObserver | None = None,
) -> tuple[float, int, dict[str, list[float]], pd.DataFrame]:
    """Run model training with automatic retry on OOM errors.

    Args:
        max_retries: Number of times to retry on OOM errors with memory cleanup.
    """
    configure_gpu_memory_growth()

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            clear_ml_memory()
            time.sleep(3 + attempt)  # Increasing delay: 4s, 5s, 6s

        try:
            _emit_phase(
                phase_observer,
                "before_preprocess_train",
                {"train_attempt": attempt + 1},
            )
            return _run_model_with_preprocessing_impl(
                train_df,
                validation_df,
                evaluation_df,
                task,
                model_name,
                model_fn,
                num_classes=num_classes,
                batch_size=batch_size,
                epochs=epochs,
                loss=loss,
                learning_rate=learning_rate,
                random_state=random_state,
                model_runtime=model_runtime,
                phase_observer=phase_observer,
            )
        except Exception as error:
            last_error = error
            if not _is_resource_exhausted_error(error):
                raise
            if attempt < max_retries:
                # Log retry attempt
                print(f"  OOM error on attempt {attempt + 1}/{max_retries + 1}, retrying after cleanup...")
                continue
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Unexpected: no result and no error")


def _run_model_with_preprocessing_impl(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    task: PreprocessingTask,
    model_name: str,
    model_fn: ModelBuilder,
    *,
    num_classes: int,
    batch_size: int,
    epochs: int,
    loss: str,
    learning_rate: float,
    random_state: int,
    model_runtime: ModelRuntimeConfig | None = None,
    phase_observer: PhaseObserver | None = None,
) -> tuple[float, int, dict[str, list[float]], pd.DataFrame]:
    enforce_reproducibility(random_state)

    clear_ml_memory()

    x_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    x_validation: np.ndarray | None = None
    y_validation: np.ndarray | None = None
    x_evaluation: np.ndarray | None = None
    train_inputs: np.ndarray | None = None
    validation_inputs: np.ndarray | None = None
    evaluation_inputs: np.ndarray | None = None
    y_train_encoded: np.ndarray | None = None
    y_validation_encoded: np.ndarray | None = None
    history: Any | None = None
    history_dict: dict[str, list[float]] | None = None
    predictions_df: pd.DataFrame | None = None
    callbacks: list[object] | None = None
    input_shape: tuple[int, ...] | None = None
    effective_batch_size = batch_size
    model = None
    try:
        x_train, y_train = preprocess_images(train_df, task.apply)
        x_validation, y_validation = preprocess_images(validation_df, task.apply)
        _emit_phase(phase_observer, "after_preprocess_train")

        _emit_phase(phase_observer, "before_prepare_train_inputs")
        train_inputs, input_shape, effective_batch_size = _prepare_model_inputs(
            x_train,
            model_name,
            batch_size,
            model_runtime,
        )
        validation_inputs, _, _ = _prepare_model_inputs(
            x_validation,
            model_name,
            batch_size,
            model_runtime,
        )
        x_train = None
        x_validation = None

        y_train_encoded = _one_hot_encode(y_train, num_classes)
        y_validation_encoded = _one_hot_encode(y_validation, num_classes)
        y_train = None
        y_validation = None
        _emit_phase(phase_observer, "after_prepare_train_inputs")

        _emit_phase(phase_observer, "before_model_build")
        model = model_fn(
            input_shape=input_shape,
            num_classes=num_classes,
            loss=loss,
            learning_rate=learning_rate,
            runtime=model_runtime,
        )
        _emit_phase(phase_observer, "after_model_build")

        _emit_phase(phase_observer, "before_fit")
        callbacks = _build_fit_callbacks()
        history = model.fit(
            train_inputs,
            y_train_encoded,
            validation_data=(validation_inputs, y_validation_encoded),
            epochs=epochs,
            batch_size=effective_batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        _emit_phase(phase_observer, "after_fit")

        train_inputs = None
        validation_inputs = None
        y_train_encoded = None
        y_validation_encoded = None
        callbacks = None
        clear_ml_memory(clear_session=False)
        _emit_phase(phase_observer, "after_train_release")

        history_dict = dict(history.history)
        history = None
        val_accuracies = history_dict.get("val_accuracy")
        if not val_accuracies:
            raise ValueError("Model history does not contain val_accuracy, " "required for model selection.")
        best_epoch = int(np.argmax(val_accuracies)) + 1
        best_val_acc = float(np.max(val_accuracies))

        _emit_phase(phase_observer, "before_eval_preprocess")
        x_evaluation, _ = preprocess_images(evaluation_df, task.apply)
        evaluation_inputs, _, _ = _prepare_model_inputs(
            x_evaluation,
            model_name,
            batch_size,
            model_runtime,
        )
        x_evaluation = None
        _emit_phase(phase_observer, "after_eval_preprocess")
        _emit_phase(phase_observer, "before_predict")
        predictions_df = _predict_dataframe(model, evaluation_inputs, evaluation_df, effective_batch_size)
        _emit_phase(phase_observer, "after_predict")

        evaluation_inputs = None
        clear_ml_memory(clear_session=False)

        return best_val_acc, best_epoch, history_dict, predictions_df
    finally:
        x_train = None
        y_train = None
        x_validation = None
        y_validation = None
        x_evaluation = None
        train_inputs = None
        validation_inputs = None
        evaluation_inputs = None
        y_train_encoded = None
        y_validation_encoded = None
        history = None
        callbacks = None
        model = None
        _clear_keras_session()


def _artifact_paths(
    history_dir: Path,
    predictions_dir: Path,
    preproc_id: str,
    model_name: str,
    param_id: str,
    augmentations_per_image: int,
) -> tuple[Path, Path]:
    suffix = f"{param_id}__aug{augmentations_per_image}"
    history_path = history_dir / preproc_id / model_name / f"history_{suffix}.csv"
    predictions_path = predictions_dir / preproc_id / model_name / f"{suffix}.csv"
    return history_path, predictions_path


def artifact_paths_for_task(config: TrainingConfig, task: TrainingTask) -> tuple[Path, Path]:
    return _artifact_paths(
        config.history_dir,
        config.predictions_dir,
        task.preproc_id,
        task.model_name,
        task.param_id,
        task.augmentations_per_image,
    )


def check_if_model_exists(
    history_dir: Path,
    predictions_dir: Path,
    preproc_id: str,
    model_name: str,
    param_id: str,
    augmentations_per_image: int,
) -> bool:
    history_path, predictions_path = _artifact_paths(
        history_dir,
        predictions_dir,
        preproc_id,
        model_name,
        param_id,
        augmentations_per_image,
    )
    return history_path.exists() and predictions_path.exists()


def task_has_existing_artifacts(config: TrainingConfig, task: TrainingTask) -> bool:
    return check_if_model_exists(
        config.history_dir,
        config.predictions_dir,
        task.preproc_id,
        task.model_name,
        task.param_id,
        task.augmentations_per_image,
    )


def _metrics_at_best_epoch(result: TrainingRunResult) -> dict[str, Any]:
    best_epoch_index = result.best_epoch - 1
    return {key: values[best_epoch_index] for key, values in result.history_dict.items() if isinstance(values, list) and best_epoch_index < len(values)}


def build_history_row(result: TrainingRunResult, config: TrainingConfig) -> dict[str, Any]:
    return {
        "fold": result.fold,
        "best_val_acc": result.best_val_acc,
        "best_epoch": result.best_epoch,
        "preproc_id": result.preproc_id,
        "model_name": result.model_name,
        "param_id": result.param_id,
        "param_combo": result.param_combo,
        "param_json": result.param_json,
        "selection_strategy": result.selection_strategy,
        "train_samples": result.train_samples,
        "validation_samples": result.validation_samples,
        "test_samples": result.test_samples,
        "augmentations_per_image": result.augmentations_per_image,
        "random_state": config.random_state,
        **_metrics_at_best_epoch(result),
        **result.summary_metrics,
    }


def save_run_result(result: TrainingRunResult, config: TrainingConfig) -> tuple[Path, Path]:
    history_path, predictions_path = _artifact_paths(
        config.history_dir,
        config.predictions_dir,
        result.preproc_id,
        result.model_name,
        result.param_id,
        result.augmentations_per_image,
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([build_history_row(result, config)]).to_csv(history_path, index=False)

    predictions_df = result.predictions_df
    predictions_df["preproc_id"] = result.preproc_id
    predictions_df["model_name"] = result.model_name
    predictions_df["param_id"] = result.param_id
    predictions_df["param_combo"] = result.param_combo
    predictions_df.to_csv(predictions_path, index=False)
    return history_path, predictions_path


def _build_validation_split(train_df: pd.DataFrame, config: TrainingConfig) -> DatasetSplitResult:
    if not 0 < config.validation_size < 1:
        raise ValueError(f"validation_size must be between 0 and 1, got {config.validation_size}.")
    return split_dataset(
        train_df,
        test_size=config.validation_size,
        random_state=config.random_state,
        label_column="label",
        group_column="split_group_id",
    )


def _filter_train_dataframe_for_augmentations(
    train_df: pd.DataFrame,
    *,
    augmentations_per_image: int,
) -> pd.DataFrame:
    if augmentations_per_image < 0:
        raise ValueError(
            "augmentations_per_image must be >= 0, "
            f"got {augmentations_per_image}."
        )
    if "is_augmented" not in train_df.columns or "augmentation_index" not in train_df.columns:
        return train_df

    augmented_mask = train_df["is_augmented"].astype(bool)
    augmentation_index = pd.to_numeric(train_df["augmentation_index"], errors="coerce").fillna(-1).astype(int)
    keep_mask = (~augmented_mask) | (augmentation_index < augmentations_per_image)
    filtered = train_df.loc[keep_mask].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            "Filtered training split is empty after applying "
            f"augmentations_per_image={augmentations_per_image}."
        )
    return filtered


def _run_fixed_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: TrainingTask,
    model_name: str,
    model_fn: ModelBuilder,
    config: TrainingConfig,
    phase_observer: PhaseObserver | None = None,
) -> TrainingRunResult:
    filtered_train_df = _filter_train_dataframe_for_augmentations(
        train_df,
        augmentations_per_image=task.augmentations_per_image,
    )
    validation_split = _build_validation_split(filtered_train_df, config)
    model_runtime = (config.model_runtime or {}).get(model_name)
    best_val_acc, best_epoch, history_dict, predictions_df = run_model_with_preprocessing(
        validation_split.train_df,
        validation_split.test_df,
        test_df,
        task.preprocessing_task,
        model_name,
        model_fn,
        num_classes=config.num_classes,
        batch_size=config.batch_size,
        epochs=config.epochs,
        loss=config.loss,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
        model_runtime=model_runtime,
        phase_observer=phase_observer,
    )
    return TrainingRunResult(
        preproc_id=task.preproc_id,
        model_name=model_name,
        param_id=task.param_id,
        param_combo=task.param_display,
        param_json=task.param_json,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        fold=None,
        history_dict=history_dict,
        predictions_df=predictions_df,
        selection_strategy="holdout_validation",
        train_samples=len(validation_split.train_df),
        validation_samples=len(validation_split.test_df),
        test_samples=len(test_df),
        augmentations_per_image=task.augmentations_per_image,
        summary_metrics={"validation_split_strategy": validation_split.strategy},
    )


def _build_cv_indices(
    train_df: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[Iterable[tuple[np.ndarray, np.ndarray]], str]:
    groups = train_df["split_group_id"] if "split_group_id" in train_df.columns else None
    labels = train_df["label"]

    if groups is not None and groups.nunique() >= config.folds:
        splitter = StratifiedGroupKFold(
            n_splits=config.folds,
            shuffle=True,
            random_state=config.random_state,
        )
        return splitter.split(train_df, labels, groups), "stratified_group_kfold"

    splitter = StratifiedKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.random_state,
    )
    return splitter.split(train_df["image_path"], labels), "stratified_kfold"


def _run_cross_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: TrainingTask,
    model_name: str,
    model_fn: ModelBuilder,
    config: TrainingConfig,
    phase_observer: PhaseObserver | None = None,
) -> TrainingRunResult:
    filtered_train_df = _filter_train_dataframe_for_augmentations(
        train_df,
        augmentations_per_image=task.augmentations_per_image,
    )
    split_iterator, cv_strategy = _build_cv_indices(filtered_train_df, config)

    best_result: TrainingRunResult | None = None
    fold_scores: list[float] = []
    fold_epochs: list[int] = []
    model_runtime = (config.model_runtime or {}).get(model_name)
    for fold_index, (fit_idx, validation_idx) in enumerate(split_iterator, start=1):
        clear_ml_memory()
        time.sleep(1)

        fit_df: pd.DataFrame | None = None
        validation_df: pd.DataFrame | None = None
        current: TrainingRunResult | None = None
        history_dict: dict[str, list[float]] | None = None
        predictions_df: pd.DataFrame | None = None
        try:
            fit_df = filtered_train_df.iloc[fit_idx].reset_index(drop=True)
            validation_df = filtered_train_df.iloc[validation_idx].reset_index(drop=True)
            fold_seed = config.random_state + fold_index
            best_val_acc, best_epoch, history_dict, predictions_df = run_model_with_preprocessing(
                fit_df,
                validation_df,
                test_df,
                task.preprocessing_task,
                model_name,
                model_fn,
                num_classes=config.num_classes,
                batch_size=config.batch_size,
                epochs=config.epochs,
                loss=config.loss,
                learning_rate=config.learning_rate,
                random_state=fold_seed,
                model_runtime=model_runtime,
                phase_observer=phase_observer,
            )

            fold_scores.append(best_val_acc)
            fold_epochs.append(best_epoch)
            current = TrainingRunResult(
                preproc_id=task.preproc_id,
                model_name=model_name,
                param_id=task.param_id,
                param_combo=task.param_display,
                param_json=task.param_json,
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
                fold=fold_index,
                history_dict=history_dict,
                predictions_df=predictions_df,
                selection_strategy="cross_validation",
                train_samples=len(fit_df),
                validation_samples=len(validation_df),
                test_samples=len(test_df),
                augmentations_per_image=task.augmentations_per_image,
            )
            if best_result is None or current.best_val_acc > best_result.best_val_acc:
                best_result = current
            else:
                current = None
                history_dict = None
                predictions_df = None
        finally:
            fit_df = None
            validation_df = None
            if current is None:
                history_dict = None
                predictions_df = None
            clear_ml_memory()

    if best_result is None:
        raise RuntimeError("Cross-validation did not produce any runs.")

    return replace(
        best_result,
        summary_metrics={
            "cv_strategy": cv_strategy,
            "cv_mean_val_acc": float(np.mean(fold_scores)),
            "cv_std_val_acc": float(np.std(fold_scores)),
            "cv_mean_best_epoch": float(np.mean(fold_epochs)),
            "cv_folds": config.folds,
        },
    )


def build_training_tasks(
    config: TrainingConfig,
    *,
    model_builders: dict[str, ModelBuilder] | None = None,
    preprocessing_tasks: Iterable[PreprocessingTask] | None = None,
) -> list[TrainingTask]:
    return list(
        iter_training_tasks(
            config,
            model_builders=model_builders,
            preprocessing_tasks=preprocessing_tasks,
        )
    )


def iter_training_tasks(
    config: TrainingConfig,
    *,
    model_builders: dict[str, ModelBuilder] | None = None,
    preprocessing_tasks: Iterable[PreprocessingTask] | None = None,
) -> Iterable[TrainingTask]:
    available_builders = model_builders or MODEL_BUILDERS
    model_names = config.model_names or list(available_builders.keys())
    resolved_preprocessing_tasks = (
        list(preprocessing_tasks)
        if preprocessing_tasks is not None
        else list(
            iter_preprocessing_tasks(
                config.preprocessing_ids,
                include_combinations=config.include_combinations,
                param_grids=config.preprocessing_grids,
            )
        )
    )

    augmentation_values = tuple(int(value) for value in config.augmentation_values)
    for augmentation_value in augmentation_values:
        if augmentation_value < 0:
            raise ValueError(
                "augmentation_values must contain integers >= 0, "
                f"got {augmentation_values!r}."
            )
        for preprocessing_task in resolved_preprocessing_tasks:
            for model_name in model_names:
                if model_name not in available_builders:
                    raise ValueError(f"Unknown model name requested: {model_name}")
                yield TrainingTask(
                    model_name=model_name,
                    preprocessing_task=preprocessing_task,
                    augmentations_per_image=augmentation_value,
                )


def run_training_task(
    task: TrainingTask,
    config: TrainingConfig,
    *,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    model_builders: dict[str, ModelBuilder] | None = None,
    phase_observer: PhaseObserver | None = None,
) -> TrainingRunResult:
    available_builders = model_builders or MODEL_BUILDERS
    if task.model_name not in available_builders:
        raise ValueError(f"Unknown model name requested: {task.model_name}")

    resolved_train_df = load_split_dataframe(config.train_split_path) if train_df is None else train_df
    resolved_test_df = load_split_dataframe(config.test_split_path) if test_df is None else test_df
    model_fn = available_builders[task.model_name]

    if config.folds == 0:
        return _run_fixed_split(
            resolved_train_df,
            resolved_test_df,
            task,
            task.model_name,
            model_fn,
            config,
            phase_observer=phase_observer,
        )
    return _run_cross_validation(
        resolved_train_df,
        resolved_test_df,
        task,
        task.model_name,
        model_fn,
        config,
        phase_observer=phase_observer,
    )


def run_training_pipeline(
    config: TrainingConfig,
    *,
    model_builders: dict[str, ModelBuilder] | None = None,
    preprocessing_tasks: Iterable[PreprocessingTask] | None = None,
) -> list[TrainingRunResult]:
    config.history_dir.mkdir(parents=True, exist_ok=True)
    config.predictions_dir.mkdir(parents=True, exist_ok=True)
    enforce_reproducibility(config.random_state)

    available_builders = model_builders or MODEL_BUILDERS
    train_df = load_split_dataframe(config.train_split_path)
    test_df = load_split_dataframe(config.test_split_path)
    execution_queue = build_training_tasks(
        config,
        model_builders=available_builders,
        preprocessing_tasks=preprocessing_tasks,
    )

    summaries: list[TrainingRunResult] = []
    current_preproc_id: str | None = None
    for task in execution_queue:
        if task.preproc_id != current_preproc_id:
            current_preproc_id = task.preproc_id
            print(f"\n=== Preprocessing: {task.preproc_id} ({task.param_display}) ===")

        print(f"--> Current Model: {task.model_name} <--")
        if config.run_skip and task_has_existing_artifacts(config, task):
            print(f"Skipping {task.label} because artifacts already exist.")
            continue

        model_started_at = time.time()
        try:
            clear_ml_memory()
            result = run_training_task(
                task,
                config,
                train_df=train_df,
                test_df=test_df,
                model_builders=available_builders,
            )
        except Exception as error:
            prefix = "GPU memory error detected for" if _is_resource_exhausted_error(error) else "Error running"
            print(f"{prefix} {task.label}: {error}")
            clear_ml_memory()
            continue

        history_path, predictions_path = save_run_result(result, config)
        summaries.append(result)
        elapsed = time.time() - model_started_at
        fold_label = f" fold {result.fold}" if result.fold else ""
        print(f"Saved best run for {task.label}{fold_label} with val_accuracy=" f"{result.best_val_acc:.4f} at epoch {result.best_epoch}.")
        print(f"History: {history_path}")
        print(f"Predictions: {predictions_path}")
        print(f"Time taken: {format_duration(elapsed)}\n")
        clear_ml_memory()

    return summaries
