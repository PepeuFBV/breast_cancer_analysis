from __future__ import annotations

import json
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

from pipeline.experiments import IterativeExperimentRunner
from pipeline.train.preprocessing import PreprocessingTask
from pipeline.train.runner import (
    TrainingConfig,
    TrainingRunResult,
    TrainingTask,
    run_model_with_preprocessing,
    run_training_task,
)
from pipeline.utils.paths import build_project_paths


class _FakeHistory:
    history = {
        "accuracy": [0.7],
        "val_accuracy": [0.8],
        "loss": [0.5],
        "val_loss": [0.4],
    }


class _FakeModel:
    def __init__(self, *, fail_on_fit: bool = False) -> None:
        self.fail_on_fit = fail_on_fit

    def fit(self, *args, **kwargs):
        if self.fail_on_fit:
            raise RuntimeError("synthetic fit failure")
        return _FakeHistory()

    def predict(self, model_inputs, batch_size=8, verbose=0):
        probabilities = np.zeros((len(model_inputs), 8), dtype="float32")
        probabilities[:, 0] = 0.75
        probabilities[:, 1] = 0.25
        return probabilities


def _write_dataset(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    image_dir = root / "images"
    image_dir.mkdir()
    image_paths: list[str] = []
    labels: list[str] = []
    for index in range(8):
        image_path = image_dir / f"image_{index}.png"
        cv2.imwrite(str(image_path), np.full((16, 16), index * 10, dtype="uint8"))
        image_paths.append(str(image_path))
        labels.append("1" if index in {0, 1, 2, 6} else "2")

    train_df = pd.DataFrame({"image_path": image_paths[:6], "label": [0, 0, 0, 1, 1, 1]})
    test_df = pd.DataFrame({"image_path": image_paths[6:], "label": [0, 1]})
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    pd.DataFrame({"image_path": image_paths[:6], "label": labels[:6]}).to_csv(
        train_csv, index=False
    )
    pd.DataFrame({"image_path": image_paths[6:], "label": labels[6:]}).to_csv(
        test_csv, index=False
    )
    return train_df, test_df, train_csv, test_csv


def _preprocessing_task() -> PreprocessingTask:
    return PreprocessingTask(
        preproc_id="none",
        params={},
        param_display="default",
        param_id="default",
        param_json="{}",
        is_combined=False,
        apply=lambda image: image,
    )


def _config(tmp_path: Path, folds: int = 0) -> TrainingConfig:
    project_paths = build_project_paths(tmp_path / "raw-data", tmp_path / "artifacts")
    project_paths.ensure_artifact_dirs()
    _, _, train_csv, test_csv = _write_dataset(tmp_path)
    return TrainingConfig(
        train_split_path=train_csv,
        test_split_path=test_csv,
        history_dir=project_paths.history_dir,
        predictions_dir=project_paths.predictions_dir,
        folds=folds,
        validation_size=0.5,
        epochs=1,
        batch_size=2,
        model_names=["custom cnn"],
        include_combinations=False,
        run_skip=False,
    )


def _fake_result(task_name: str, *, param_id: str = "default") -> TrainingRunResult:
    return TrainingRunResult(
        preproc_id=task_name,
        model_name="custom cnn",
        param_id=param_id,
        param_combo=task_name,
        param_json=json.dumps({"task": task_name}, sort_keys=True),
        best_val_acc=0.8,
        best_epoch=1,
        fold=None,
        history_dict={"accuracy": [0.7], "val_accuracy": [0.8]},
        predictions_df=pd.DataFrame(
            {
                "y_true": [0, 1],
                "y_pred": [0, 1],
                "y_pred_probability": [[0.8, 0.2], [0.1, 0.9]],
                "prob_class_0": [0.8, 0.1],
                "prob_class_1": [0.2, 0.9],
            }
        ),
        selection_strategy="holdout_validation",
        train_samples=6,
        validation_samples=2,
        test_samples=2,
    )


def test_run_model_with_preprocessing_cleans_on_success(tmp_path) -> None:
    train_df, test_df, _, _ = _write_dataset(tmp_path)
    cleanup_calls: list[dict[str, object]] = []

    with patch(
        "pipeline.train.runner.clear_ml_memory",
        side_effect=lambda **kwargs: cleanup_calls.append(kwargs),
    ):
        result = run_model_with_preprocessing(
            train_df.iloc[:4].reset_index(drop=True),
            train_df.iloc[4:].reset_index(drop=True),
            test_df.reset_index(drop=True),
            _preprocessing_task(),
            "custom cnn",
            lambda **kwargs: _FakeModel(),
            num_classes=8,
            batch_size=2,
            epochs=1,
            loss="categorical_crossentropy",
            learning_rate=1e-4,
            random_state=42,
        )

    assert result[0] == 0.8
    assert any(call.get("clear_session") is False for call in cleanup_calls)
    assert any("clear_session" not in call for call in cleanup_calls)


def test_run_model_with_preprocessing_cleans_on_failure(tmp_path) -> None:
    train_df, test_df, _, _ = _write_dataset(tmp_path)
    cleanup_calls: list[dict[str, object]] = []

    with patch(
        "pipeline.train.runner.clear_ml_memory",
        side_effect=lambda **kwargs: cleanup_calls.append(kwargs),
    ):
        with pytest.raises(RuntimeError, match="synthetic fit failure"):
            run_model_with_preprocessing(
                train_df.iloc[:4].reset_index(drop=True),
                train_df.iloc[4:].reset_index(drop=True),
                test_df.reset_index(drop=True),
                _preprocessing_task(),
                "custom cnn",
                lambda **kwargs: _FakeModel(fail_on_fit=True),
                num_classes=8,
                batch_size=2,
                epochs=1,
                loss="categorical_crossentropy",
                learning_rate=1e-4,
                random_state=42,
            )

    assert len(cleanup_calls) >= 2
    assert any("clear_session" not in call for call in cleanup_calls)


def test_cross_validation_cleans_once_per_fold(tmp_path) -> None:
    config = _config(tmp_path, folds=3)
    cleanup_calls: list[dict[str, object]] = []
    task = TrainingTask(model_name="custom cnn", preprocessing_task=_preprocessing_task())

    with patch(
        "pipeline.train.runner.run_model_with_preprocessing",
        return_value=(
            0.8,
            1,
            {"accuracy": [0.7], "val_accuracy": [0.8]},
            pd.DataFrame(
                {
                    "y_true": [0],
                    "y_pred": [0],
                    "y_pred_probability": [[1.0]],
                    "prob_class_0": [1.0],
                }
            ),
        ),
    ):
        with patch(
            "pipeline.train.runner.clear_ml_memory",
            side_effect=lambda **kwargs: cleanup_calls.append(kwargs),
        ):
            result = run_training_task(
                task=task,
                config=config,
                model_builders={"custom cnn": lambda **kwargs: _FakeModel()},
            )

    assert result.fold in {1, 2, 3}
    assert len(cleanup_calls) >= config.folds * 2


def test_iterative_runner_cleans_after_failed_task(tmp_path) -> None:
    config = _config(tmp_path)
    project_paths = build_project_paths(tmp_path / "raw-data", tmp_path / "artifacts")
    project_paths.ensure_artifact_dirs()
    runner = IterativeExperimentRunner(
        config_path=Path("configs/experiment.smoke.json"),
        project_paths=project_paths,
        training_config=config,
        preprocessing_tasks=[_preprocessing_task()],
    )
    cleanup_calls: list[dict[str, object]] = []

    with patch(
        "pipeline.experiments.runner.run_training_task",
        side_effect=RuntimeError("synthetic runner failure"),
    ):
        with patch(
            "pipeline.experiments.runner.clear_ml_memory",
            side_effect=lambda **kwargs: cleanup_calls.append(kwargs),
        ):
            snapshot = runner.run()

    assert snapshot["counts"]["failed"] == 1
    assert len(cleanup_calls) >= 2


@pytest.mark.memory
def test_mocked_long_queue_memory_stays_bounded(tmp_path) -> None:
    config = _config(tmp_path)
    project_paths = build_project_paths(tmp_path / "raw-data", tmp_path / "artifacts")
    project_paths.ensure_artifact_dirs()
    runner = IterativeExperimentRunner(
        config_path=Path("configs/experiment.smoke.json"),
        project_paths=project_paths,
        training_config=config,
        preprocessing_tasks=[
            PreprocessingTask(
                preproc_id=f"task_{index:02d}",
                params={"index": index},
                param_display=f"index={index}",
                param_id=f"task-{index}",
                param_json=json.dumps({"index": index}, sort_keys=True),
                is_combined=False,
                apply=lambda image: image,
            )
            for index in range(20)
        ],
    )

    def fake_run_training_task(task, *args, **kwargs):
        payload = [bytearray(250_000) for _ in range(2)]
        payload = None
        return _fake_result(task.preproc_id, param_id=task.param_id)

    tracemalloc.start()
    with patch(
        "pipeline.experiments.runner.run_training_task",
        side_effect=fake_run_training_task,
    ):
        snapshot = runner.run()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert snapshot["counts"]["completed"] == 20
    assert current < 6_000_000
    assert peak > current
