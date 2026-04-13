from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline.experiments import (
    ExperimentStateStore,
    IterativeExperimentRunner,
    IterativeRunOptions,
)
from pipeline.train.preprocessing import PreprocessingTask
from pipeline.train.runner import TrainingConfig
from pipeline.utils.paths import build_project_paths


class _FakeHistory:
    def __init__(self, value: float) -> None:
        self.history = {
            "accuracy": [value - 0.1, value - 0.05],
            "val_accuracy": [value - 0.05, value],
        }


class _CountingModel:
    def __init__(
        self,
        value: float,
        *,
        on_fit=None,
        fail_on_fit: bool = False,
    ) -> None:
        self.value = value
        self.on_fit = on_fit
        self.fail_on_fit = fail_on_fit

    def fit(self, *args, **kwargs):
        if self.on_fit is not None:
            self.on_fit()
        if self.fail_on_fit:
            raise RuntimeError("synthetic fit failure")
        return _FakeHistory(self.value)

    def predict(self, model_inputs, batch_size=8, verbose=0):
        probabilities = np.zeros((len(model_inputs), 8), dtype="float32")
        probabilities[:, 0] = 0.8
        probabilities[:, 1] = 0.2
        return probabilities


def _write_dataset(root: Path) -> tuple[Path, Path]:
    image_dir = root / "images"
    image_dir.mkdir()

    image_paths: list[str] = []
    labels: list[str] = []
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
    return train_path, test_path


def _build_training_config(root: Path) -> tuple[TrainingConfig, object]:
    project_paths = build_project_paths(root / "raw-data", root / "artifacts")
    project_paths.ensure_artifact_dirs()
    train_path, test_path = _write_dataset(root)
    config = TrainingConfig(
        train_split_path=train_path,
        test_split_path=test_path,
        history_dir=project_paths.history_dir,
        predictions_dir=project_paths.predictions_dir,
        folds=0,
        validation_size=0.5,
        epochs=2,
        batch_size=2,
        learning_rate=1e-4,
        loss="categorical_crossentropy",
        model_names=["custom cnn"],
        include_combinations=False,
    )
    return config, project_paths


def _task(name: str, params: dict[str, object] | None = None) -> PreprocessingTask:
    resolved_params = params or {}
    param_json = json.dumps(resolved_params, sort_keys=True, separators=(",", ":"))
    return PreprocessingTask(
        preproc_id=name,
        params=resolved_params,
        param_display="default" if not resolved_params else param_json,
        param_id="default" if not resolved_params else f"{name}-params",
        param_json=param_json,
        is_combined=False,
        apply=lambda image: image,
    )


class IterativeRunnerTest(unittest.TestCase):
    def test_runner_resumes_without_rerunning_completed_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            fit_calls = {"count": 0}

            def fake_builder(*args, **kwargs):
                def on_fit():
                    fit_calls["count"] += 1

                return _CountingModel(0.87, on_fit=on_fit)

            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": fake_builder},
                preprocessing_tasks=[_task("none")],
            )

            first_snapshot = runner.run()
            second_snapshot = runner.run()

            self.assertEqual(fit_calls["count"], 1)
            self.assertEqual(first_snapshot["counts"]["completed"], 1)
            self.assertEqual(second_snapshot["counts"]["completed"], 1)

            store = ExperimentStateStore(project_paths)
            state = store.load_state()
            self.assertEqual(len(state["tasks"]), 1)
            self.assertEqual(state["tasks"][0]["status"], "completed")
            self.assertEqual(state["tasks"][0]["attempts"], 1)
            self.assertTrue(store.state_path.exists())
            self.assertTrue(store.summary_path.exists())

    def test_stop_request_waits_for_current_task_and_resumes_pending_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            store = ExperimentStateStore(project_paths)
            fit_calls = {"count": 0}

            def fake_builder(*args, **kwargs):
                def on_fit():
                    fit_calls["count"] += 1
                    if fit_calls["count"] == 1:
                        store.request_stop(reason="test-stop")

                return _CountingModel(0.82, on_fit=on_fit)

            tasks = [_task("none"), _task("none", {"variant": "second"})]
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": fake_builder},
                preprocessing_tasks=tasks,
            )

            first_snapshot = runner.run()
            second_snapshot = runner.run()

            self.assertEqual(fit_calls["count"], 2)
            self.assertEqual(first_snapshot["counts"]["completed"], 1)
            self.assertEqual(first_snapshot["counts"]["pending"], 1)
            self.assertTrue(first_snapshot["stop_requested"])
            self.assertEqual(second_snapshot["counts"]["completed"], 2)
            self.assertEqual(second_snapshot["counts"]["pending"], 0)

    def test_failed_tasks_require_explicit_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            fit_calls = {"count": 0}

            def fake_builder(*args, **kwargs):
                fit_calls["count"] += 1
                return _CountingModel(
                    0.9,
                    fail_on_fit=(fit_calls["count"] == 1),
                )

            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": fake_builder},
                preprocessing_tasks=[_task("none")],
            )

            first_snapshot = runner.run()
            second_snapshot = runner.run()
            third_snapshot = runner.run(IterativeRunOptions(rerun_failed=True))

            self.assertEqual(fit_calls["count"], 2)
            self.assertEqual(first_snapshot["counts"]["failed"], 1)
            self.assertEqual(second_snapshot["counts"]["failed"], 1)
            self.assertEqual(third_snapshot["counts"]["completed"], 1)
            self.assertEqual(third_snapshot["counts"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
