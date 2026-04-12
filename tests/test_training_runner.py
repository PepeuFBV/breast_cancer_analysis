from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline.train.preprocessing import PreprocessingTask
from pipeline.train.runner import TrainingConfig, run_training_pipeline


class _FakeHistory:
    def __init__(self, value: float) -> None:
        self.history = {
            "accuracy": [value - 0.1, value - 0.05],
            "val_accuracy": [value - 0.05, value],
        }


class _FakeModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def fit(self, *args, **kwargs):
        return _FakeHistory(self.value)

    def predict(self, model_inputs, batch_size=8, verbose=0):
        probabilities = np.zeros((len(model_inputs), 8), dtype="float32")
        probabilities[:, 0] = 0.8
        probabilities[:, 1] = 0.2
        return probabilities


class TrainingRunnerTest(unittest.TestCase):
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
                labels.append("1" if index < 4 else "2")

            train_df = pd.DataFrame({"image_path": image_paths[:4], "label": labels[:4]})
            test_df = pd.DataFrame({"image_path": image_paths[4:], "label": labels[4:]})
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            values = iter([0.71, 0.94])

            def fake_builder(input_shape, num_classes, loss="categorical_crossentropy"):
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
            history_path = root / "history" / "none" / "custom cnn" / "history_default.csv"
            predictions_path = root / "predictions" / "none" / "custom cnn" / "default.csv"
            self.assertTrue(history_path.exists())
            self.assertTrue(predictions_path.exists())

            history_df = pd.read_csv(history_path)
            self.assertEqual(int(history_df.iloc[0]["fold"]), 2)
            self.assertAlmostEqual(float(history_df.iloc[0]["best_val_acc"]), 0.94)


if __name__ == "__main__":
    unittest.main()
