from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pipeline.data.constants import DEFAULT_RANDOM_STATE, LABEL_MAPPING
from pipeline.train.models import MODEL_BUILDERS, ModelBuilder
from pipeline.train.preprocessing import PreprocessingTask, iter_preprocessing_tasks
from pipeline.utils.naming import param_dict_to_display
from pipeline.utils.runtime import format_duration


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


SINGLE_CHANNEL_MODEL_NAMES = {"custom cnn"}


@dataclass(frozen=True)
class TrainingConfig:
    train_split_path: Path
    test_split_path: Path
    history_dir: Path
    predictions_dir: Path
    folds: int = 4
    batch_size: int = 8
    epochs: int = 15
    num_classes: int = len(LABEL_MAPPING)
    run_skip: bool = True
    random_state: int = DEFAULT_RANDOM_STATE
    model_names: list[str] | None = None
    preprocessing_ids: list[str] | None = None
    include_combinations: bool = True


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


def load_split_dataframe(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    raw_labels = dataframe["label"].astype(str).str.strip().str.lower()
    mapped_labels = raw_labels.map(LABEL_MAPPING)
    if mapped_labels.isna().any():
        invalid_rows = raw_labels[mapped_labels.isna()]
        raise ValueError(f"Invalid labels found in split {path}: {sorted(invalid_rows.unique())}")
    dataframe["label"] = mapped_labels.astype(int)
    return dataframe


def merge_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_marked = train_df.copy()
    test_marked = test_df.copy()
    train_marked["set"] = "train"
    test_marked["set"] = "test"
    return pd.concat([train_marked, test_marked], ignore_index=True)


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


def _prepare_model_inputs(images: np.ndarray, model_name: str, batch_size: int) -> tuple[np.ndarray, tuple[int, ...], int]:
    images = np.expand_dims(images, -1).astype("float32") / 255.0
    if model_name.lower() in SINGLE_CHANNEL_MODEL_NAMES:
        return images, tuple(images.shape[1:]), batch_size

    images = np.repeat(images, 3, axis=-1)
    return images, tuple(images.shape[1:]), 4


def _one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype="float32")[labels]


def _is_resource_exhausted_error(error: Exception) -> bool:
    return error.__class__.__name__ == "ResourceExhaustedError"


def _clear_keras_session() -> None:
    try:
        from keras import backend as backend
    except Exception:
        return
    backend.clear_session()


def _predict_dataframe(
    model: object,
    model_inputs: np.ndarray,
    test_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    probabilities = model.predict(model_inputs, batch_size=batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    prediction_df = pd.DataFrame(
        {
            "y_true": test_df["label"].values,
            "y_pred": predictions,
            "y_pred_probability": probabilities.tolist(),
        }
    )
    for index in range(probabilities.shape[1]):
        prediction_df[f"prob_class_{index}"] = probabilities[:, index]
    return prediction_df


def run_model_with_preprocessing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: PreprocessingTask,
    model_name: str,
    model_fn: ModelBuilder,
    *,
    num_classes: int,
    batch_size: int,
    epochs: int,
) -> tuple[float, int, dict[str, list[float]], pd.DataFrame]:
    x_train, y_train = preprocess_images(train_df, task.apply)
    x_test, y_test = preprocess_images(test_df, task.apply)

    train_inputs, input_shape, effective_batch_size = _prepare_model_inputs(x_train, model_name, batch_size)
    test_inputs, _, _ = _prepare_model_inputs(x_test, model_name, batch_size)
    y_train_encoded = _one_hot_encode(y_train, num_classes)
    y_test_encoded = _one_hot_encode(y_test, num_classes)

    model = model_fn(input_shape=input_shape, num_classes=num_classes, loss="categorical_crossentropy")
    try:
        history = model.fit(
            train_inputs,
            y_train_encoded,
            validation_data=(test_inputs, y_test_encoded),
            epochs=epochs,
            batch_size=effective_batch_size,
            verbose=0,
        )
        history_dict = history.history
        val_accuracies = history_dict["val_accuracy"]
        best_epoch = int(np.argmax(val_accuracies)) + 1
        best_val_acc = float(np.max(val_accuracies))
        predictions_df = _predict_dataframe(model, test_inputs, test_df, effective_batch_size)
        return best_val_acc, best_epoch, history_dict, predictions_df
    finally:
        _clear_keras_session()


def _artifact_paths(history_dir: Path, predictions_dir: Path, preproc_id: str, model_name: str, param_id: str) -> tuple[Path, Path]:
    history_path = history_dir / preproc_id / model_name / f"history_{param_id}.csv"
    predictions_path = predictions_dir / preproc_id / model_name / f"{param_id}.csv"
    return history_path, predictions_path


def check_if_model_exists(history_dir: Path, predictions_dir: Path, preproc_id: str, model_name: str, param_id: str) -> bool:
    history_path, predictions_path = _artifact_paths(history_dir, predictions_dir, preproc_id, model_name, param_id)
    return history_path.exists() and predictions_path.exists()


def _save_run_result(result: TrainingRunResult, config: TrainingConfig) -> tuple[Path, Path]:
    history_path, predictions_path = _artifact_paths(
        config.history_dir,
        config.predictions_dir,
        result.preproc_id,
        result.model_name,
        result.param_id,
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    best_epoch_index = result.best_epoch - 1
    metrics = {
        key: values[best_epoch_index]
        for key, values in result.history_dict.items()
        if isinstance(values, list)
    }

    history_row = {
        "fold": result.fold,
        "best_val_acc": result.best_val_acc,
        "best_epoch": result.best_epoch,
        "preproc_id": result.preproc_id,
        "model_name": result.model_name,
        "param_id": result.param_id,
        "param_combo": result.param_combo,
        "param_json": result.param_json,
        **metrics,
    }
    pd.DataFrame([history_row]).to_csv(history_path, index=False)

    predictions_df = result.predictions_df.copy()
    predictions_df["preproc_id"] = result.preproc_id
    predictions_df["model_name"] = result.model_name
    predictions_df["param_id"] = result.param_id
    predictions_df["param_combo"] = result.param_combo
    predictions_df.to_csv(predictions_path, index=False)
    return history_path, predictions_path


def _run_fixed_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: PreprocessingTask,
    model_name: str,
    model_fn: ModelBuilder,
    config: TrainingConfig,
) -> TrainingRunResult:
    best_val_acc, best_epoch, history_dict, predictions_df = run_model_with_preprocessing(
        train_df,
        test_df,
        task,
        model_name,
        model_fn,
        num_classes=config.num_classes,
        batch_size=config.batch_size,
        epochs=config.epochs,
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
    )


def _run_cross_validation(
    merged_df: pd.DataFrame,
    task: PreprocessingTask,
    model_name: str,
    model_fn: ModelBuilder,
    config: TrainingConfig,
) -> TrainingRunResult:
    splitter = StratifiedKFold(
        n_splits=config.folds,
        shuffle=True,
        random_state=config.random_state,
    )
    x = merged_df["image_path"]
    y = merged_df["label"]

    best_result: TrainingRunResult | None = None
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        train_set = merged_df.iloc[train_idx].reset_index(drop=True)
        test_set = merged_df.iloc[test_idx].reset_index(drop=True)
        best_val_acc, best_epoch, history_dict, predictions_df = run_model_with_preprocessing(
            train_set,
            test_set,
            task,
            model_name,
            model_fn,
            num_classes=config.num_classes,
            batch_size=config.batch_size,
            epochs=config.epochs,
        )

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
        )
        if best_result is None or current.best_val_acc > best_result.best_val_acc:
            best_result = current

    if best_result is None:
        raise RuntimeError("Cross-validation did not produce any runs.")
    return best_result


def run_training_pipeline(
    config: TrainingConfig,
    *,
    model_builders: dict[str, ModelBuilder] | None = None,
    preprocessing_tasks: Iterable[PreprocessingTask] | None = None,
) -> list[TrainingRunResult]:
    config.history_dir.mkdir(parents=True, exist_ok=True)
    config.predictions_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split_dataframe(config.train_split_path)
    test_df = load_split_dataframe(config.test_split_path)
    merged_df = merge_train_test(train_df, test_df) if config.folds > 0 else None

    available_builders = model_builders or MODEL_BUILDERS
    model_names = config.model_names or list(available_builders.keys())
    tasks = preprocessing_tasks or list(
        iter_preprocessing_tasks(
            config.preprocessing_ids,
            include_combinations=config.include_combinations,
        )
    )

    summaries: list[TrainingRunResult] = []
    for task in tasks:
        print(f"\n=== Preprocessing: {task.preproc_id} ({task.param_display}) ===")
        for model_name in model_names:
            model_fn = available_builders[model_name]
            print(f"--> Current Model: {model_name} <--")
            if config.run_skip and check_if_model_exists(
                config.history_dir,
                config.predictions_dir,
                task.preproc_id,
                model_name,
                task.param_id,
            ):
                print(
                    f"Skipping {task.preproc_id} [{model_name} - {task.param_display}] "
                    "because artifacts already exist."
                )
                continue

            model_started_at = time.time()
            try:
                if config.folds == 0:
                    result = _run_fixed_split(train_df, test_df, task, model_name, model_fn, config)
                else:
                    if merged_df is None:
                        raise RuntimeError("Merged dataframe was not initialized for cross-validation.")
                    result = _run_cross_validation(merged_df, task, model_name, model_fn, config)
            except Exception as error:
                if _is_resource_exhausted_error(error):
                    print(
                        "GPU memory error detected for "
                        f"{task.preproc_id} [{model_name} - {task.param_display}]. "
                        "Exiting for external restart..."
                    )
                    os._exit(1)
                print(f"Error running {task.preproc_id} [{model_name} - {task.param_display}]: {error}")
                continue

            history_path, predictions_path = _save_run_result(result, config)
            summaries.append(result)
            elapsed = time.time() - model_started_at
            fold_label = f" fold {result.fold}" if result.fold else ""
            print(
                f"Saved best run for {task.preproc_id} [{model_name} - {task.param_display}]"
                f"{fold_label} with val_accuracy={result.best_val_acc:.4f} at epoch {result.best_epoch}."
            )
            print(f"History: {history_path}")
            print(f"Predictions: {predictions_path}")
            print(f"Time taken: {format_duration(elapsed)}\n")

    return summaries
