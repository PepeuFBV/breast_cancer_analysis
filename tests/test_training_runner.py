from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline.train.models import ModelRuntimeConfig
from pipeline.train.preprocessing import PreprocessingTask
from pipeline.train.runner import (
    TrainingConfig,
    build_training_tasks,
    iter_training_tasks,
    run_training_pipeline,
)


class _FakeHistory:
    def __init__(self, value: float) -> None:
        self.history = {
            "accuracy": [value - 0.1, value - 0.05],
            "val_accuracy": [value - 0.05, value],
        }


class _FakeModel:
    def __init__(self, value: float, captures: dict | None = None) -> None:
        self.value = value
        self.captures = captures if captures is not None else {}

    def fit(self, *args, **kwargs):
        self.captures["fit_batch_size"] = kwargs.get("batch_size")
        self.captures["fit_validation_shape"] = tuple(kwargs["validation_data"][0].shape[1:])
        self.captures["fit_validation_count"] = int(kwargs["validation_data"][0].shape[0])
        return _FakeHistory(self.value)

    def predict(self, model_inputs, batch_size=8, verbose=0):
        self.captures["predict_batch_size"] = batch_size
        probabilities = np.zeros((len(model_inputs), 8), dtype="float32")
        probabilities[:, 0] = 0.8
        probabilities[:, 1] = 0.2
        return probabilities


class TrainingRunnerTest(unittest.TestCase):
    def test_build_training_tasks_expands_preprocessing_model_grid(self) -> None:
        task_a = PreprocessingTask(
            preproc_id="none",
            params={},
            param_display="default",
            param_id="default",
            param_json="{}",
            is_combined=False,
            apply=lambda image: image,
        )
        task_b = PreprocessingTask(
            preproc_id="denoise",
            params={"sigma": 1},
            param_display="sigma=1",
            param_id="sigma-1",
            param_json='{"sigma":1}',
            is_combined=False,
            apply=lambda image: image,
        )
        config = TrainingConfig(
            train_split_path=Path("train.csv"),
            test_split_path=Path("test.csv"),
            history_dir=Path("history"),
            predictions_dir=Path("predictions"),
            model_names=["custom cnn", "resnet"],
            include_combinations=False,
        )

        tasks = build_training_tasks(
            config,
            model_builders={
                "custom cnn": lambda *args, **kwargs: _FakeModel(0.8),
                "resnet": lambda *args, **kwargs: _FakeModel(0.8),
            },
            preprocessing_tasks=[task_a, task_b],
        )

        self.assertEqual(
            [(task.preproc_id, task.model_name) for task in tasks],
            [
                ("none", "custom cnn"),
                ("none", "resnet"),
                ("denoise", "custom cnn"),
                ("denoise", "resnet"),
            ],
        )

    def test_build_training_tasks_expands_augmentation_dimension(self) -> None:
        task = PreprocessingTask(
            preproc_id="none",
            params={},
            param_display="default",
            param_id="default",
            param_json="{}",
            is_combined=False,
            apply=lambda image: image,
        )
        config = TrainingConfig(
            train_split_path=Path("train.csv"),
            test_split_path=Path("test.csv"),
            history_dir=Path("history"),
            predictions_dir=Path("predictions"),
            model_names=["custom cnn"],
            include_combinations=False,
            augmentation_values=(1, 2, 3),
        )

        tasks = build_training_tasks(
            config,
            model_builders={"custom cnn": lambda *args, **kwargs: _FakeModel(0.8)},
            preprocessing_tasks=[task],
        )

        self.assertEqual([task_entry.augmentations_per_image for task_entry in tasks], [1, 2, 3])

    def test_iter_training_tasks_streams_preprocessing_iterable(self) -> None:
        task = PreprocessingTask(
            preproc_id="none",
            params={},
            param_display="default",
            param_id="default",
            param_json="{}",
            is_combined=False,
            apply=lambda image: image,
        )
        config = TrainingConfig(
            train_split_path=Path("train.csv"),
            test_split_path=Path("test.csv"),
            history_dir=Path("history"),
            predictions_dir=Path("predictions"),
            model_names=["custom cnn"],
            include_combinations=False,
            augmentation_values=(1, 2),
        )

        def _preprocessing_stream():
            yield task
            raise AssertionError("iterator should not be fully consumed before first task is yielded")

        iterator = iter_training_tasks(
            config,
            model_builders={"custom cnn": lambda *args, **kwargs: _FakeModel(0.8)},
            preprocessing_tasks=_preprocessing_stream(),
        )

        first_task = next(iter(iterator))

        self.assertEqual(first_task.preproc_id, "none")
        self.assertEqual(first_task.model_name, "custom cnn")
        self.assertEqual(first_task.augmentations_per_image, 1)

    def test_cross_validation_persists_best_fold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            image_dir.mkdir()

            image_paths = []
            labels = []
            for index in range(8):
                image_path = image_dir / f"image_{index}.png"
                cv2.imwrite(str(image_path), np.full((8, 8), index, dtype="uint8"))
                image_paths.append(str(image_path))
                labels.append("1" if index in {0, 1, 4, 5} else "2")

            train_df = pd.DataFrame({"image_path": image_paths[:4], "label": labels[:4]})
            test_df = pd.DataFrame({"image_path": image_paths[4:], "label": labels[4:]})
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            values = iter([0.71, 0.94])

            def fake_builder(
                input_shape,
                num_classes,
                loss="categorical_crossentropy",
                learning_rate=1e-4,
                runtime=None,
            ):
                return _FakeModel(next(values))

            task = PreprocessingTask(
                preproc_id="none",
                params={},
                param_display="default",
                param_id="default",
                param_json="{}",
                is_combined=False,
                apply=lambda image: image,
            )
            config = TrainingConfig(
                train_split_path=train_path,
                test_split_path=test_path,
                history_dir=root / "history",
                predictions_dir=root / "predictions",
                folds=2,
                epochs=2,
                batch_size=2,
                model_names=["custom cnn"],
                include_combinations=False,
            )

            results = run_training_pipeline(
                config,
                model_builders={"custom cnn": fake_builder},
                preprocessing_tasks=[task],
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].fold, 2)
            history_path = root / "history" / "none" / "custom cnn" / "history_default__aug1.csv"
            predictions_path = root / "predictions" / "none" / "custom cnn" / "default__aug1.csv"
            self.assertTrue(history_path.exists())
            self.assertTrue(predictions_path.exists())

            history_df = pd.read_csv(history_path)
            self.assertEqual(int(history_df.iloc[0]["fold"]), 2)
            self.assertAlmostEqual(float(history_df.iloc[0]["best_val_acc"]), 0.94)
            self.assertEqual(history_df.iloc[0]["selection_strategy"], "cross_validation")
            self.assertIn("cv_mean_val_acc", history_df.columns)

            predictions_df = pd.read_csv(predictions_path)
            self.assertEqual(len(predictions_df), len(test_df))

    def test_training_uses_configured_runtime_and_optimizer_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            image_dir.mkdir()

            image_paths = []
            labels = []
            for index in range(8):
                image_path = image_dir / f"image_{index}.png"
                cv2.imwrite(str(image_path), np.full((8, 8), index, dtype="uint8"))
                image_paths.append(str(image_path))
                labels.append("1" if index < 4 else "2")

            train_df = pd.DataFrame({"image_path": image_paths[:6], "label": labels[:6]})
            test_df = pd.DataFrame({"image_path": image_paths[6:], "label": labels[6:]})
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            captures: dict[str, object] = {}

            def fake_builder(
                input_shape,
                num_classes,
                loss="categorical_crossentropy",
                learning_rate=1e-4,
                runtime=None,
            ):
                captures["input_shape"] = input_shape
                captures["loss"] = loss
                captures["learning_rate"] = learning_rate
                captures["runtime"] = runtime
                return _FakeModel(0.83, captures)

            task = PreprocessingTask(
                preproc_id="none",
                params={},
                param_display="default",
                param_id="default",
                param_json="{}",
                is_combined=False,
                apply=lambda image: image,
            )
            config = TrainingConfig(
                train_split_path=train_path,
                test_split_path=test_path,
                history_dir=root / "history",
                predictions_dir=root / "predictions",
                folds=0,
                validation_size=0.5,
                epochs=2,
                batch_size=8,
                learning_rate=5e-4,
                loss="mean_squared_error",
                model_names=["resnet"],
                include_combinations=False,
                model_runtime={
                    "resnet": ModelRuntimeConfig(
                        input_channels=3,
                        batch_size=3,
                        dense_units=64,
                        dropout_rate=0.2,
                    )
                },
            )

            results = run_training_pipeline(
                config,
                model_builders={"resnet": fake_builder},
                preprocessing_tasks=[task],
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(captures["input_shape"], (8, 8, 3))
            self.assertEqual(captures["loss"], "mean_squared_error")
            self.assertAlmostEqual(captures["learning_rate"], 5e-4)
            self.assertEqual(captures["fit_batch_size"], 3)
            self.assertEqual(captures["predict_batch_size"], 3)
            self.assertEqual(captures["fit_validation_shape"], (8, 8, 3))
            self.assertEqual(captures["fit_validation_count"], 3)

            history_path = root / "history" / "none" / "resnet" / "history_default__aug1.csv"
            predictions_path = root / "predictions" / "none" / "resnet" / "default__aug1.csv"
            history_df = pd.read_csv(history_path)
            predictions_df = pd.read_csv(predictions_path)
            self.assertEqual(history_df.iloc[0]["selection_strategy"], "holdout_validation")
            self.assertEqual(len(predictions_df), len(test_df))


if __name__ == "__main__":
    unittest.main()
