from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from pipeline.config import load_experiment_config
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
    param_id = "default"
    if resolved_params:
        param_id = "-".join(
            [name, *[f"{key}-{value}" for key, value in resolved_params.items()]]
        )
    return PreprocessingTask(
        preproc_id=name,
        params=resolved_params,
        param_display="default" if not resolved_params else param_json,
        param_id=param_id,
        param_json=param_json,
        is_combined=False,
        apply=lambda image: image,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class IterativeRunnerTest(unittest.TestCase):
    def test_build_experiment_id_does_not_resolve_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={
                    "custom cnn": lambda *args, **kwargs: _CountingModel(0.8)
                },
                preprocessing_tasks=[_task("none")],
            )

            with patch("pathlib.Path.resolve", side_effect=AssertionError("resolve")):
                record, _task_entry = runner.build_queue()[0]
                experiment_id = record["id"]

            self.assertTrue(experiment_id.startswith("exp-"))

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

    def test_stop_request_waits_for_current_task_and_resumes_pending_tasks(
        self,
    ) -> None:
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

    def test_runner_continues_past_tenth_task_and_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            fit_calls = {"count": 0}

            def fake_builder(*args, **kwargs):
                def on_fit():
                    fit_calls["count"] += 1
                    if fit_calls["count"] == 10:
                        raise RuntimeError("synthetic tenth failure")

                return _CountingModel(0.84, on_fit=on_fit)

            tasks = [_task(f"variant_{index:02d}") for index in range(12)]
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": fake_builder},
                preprocessing_tasks=tasks,
            )

            snapshot = runner.run()

            self.assertEqual(fit_calls["count"], 12)
            self.assertEqual(snapshot["counts"]["completed"], 11)
            self.assertEqual(snapshot["counts"]["failed"], 1)

            store = ExperimentStateStore(project_paths)
            state = store.load_state()
            statuses = [task["status"] for task in state["tasks"]]
            self.assertEqual(len(statuses), 12)
            self.assertEqual(statuses[9], "failed")
            self.assertEqual(statuses[10:], ["completed", "completed"])
            self.assertIn("synthetic tenth failure", state["tasks"][9]["error_summary"])
            self.assertTrue(store.summary_path.exists())
            self.assertEqual(
                len(list(project_paths.experiment_task_dir.glob("*.json"))), 12
            )

    def test_status_reconciles_stale_running_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={
                    "custom cnn": lambda *args, **kwargs: _CountingModel(0.8)
                },
                preprocessing_tasks=[_task("none")],
            )
            record, _ = runner.build_queue()[0]
            store = ExperimentStateStore(project_paths)
            store.ensure_dirs()
            store.sync_queue(
                [record], config_path=Path("configs/experiment.default.json")
            )
            store.update_task_status(record["id"], status="running")

            snapshot = store.summarize()
            state = store.load_state()

            self.assertEqual(snapshot["counts"]["running"], 0)
            self.assertEqual(snapshot["counts"]["stopped"], 1)
            self.assertEqual(state["tasks"][0]["status"], "stopped")

    def test_runner_rejects_excessively_large_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            experiment_config = load_experiment_config()
            expanded_config = replace(
                config,
                include_combinations=True,
                preprocessing_grids=experiment_config.preprocess.preprocessing_grid,
            )
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=expanded_config,
                model_builders={
                    "custom cnn": lambda *args, **kwargs: _CountingModel(0.8)
                },
            )

            with self.assertRaises(ValueError) as context:
                runner.build_queue()

            self.assertIn("safety limit", str(context.exception))

    def test_keyboard_interrupt_marks_current_task_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={
                    "custom cnn": lambda *args, **kwargs: _CountingModel(0.8)
                },
                preprocessing_tasks=[_task("none")],
            )

            with patch(
                "pipeline.experiments.runner.run_training_task",
                side_effect=KeyboardInterrupt("manual stop"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.run()

            store = ExperimentStateStore(project_paths)
            state = store.load_state()
            self.assertEqual(state["tasks"][0]["status"], "stopped")
            self.assertIn("manual stop", state["tasks"][0]["error_summary"])
            self.assertIsNone(state["current_task_id"])
            self.assertFalse(store.pid_path.exists())

    def test_structured_logging_creates_run_and_task_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={
                    "custom cnn": lambda *args, **kwargs: _CountingModel(0.8)
                },
                preprocessing_tasks=[_task("none")],
            )

            snapshot = runner.run()
            self.assertEqual(snapshot["counts"]["completed"], 1)

            store = ExperimentStateStore(project_paths)
            run_events_path = store.run_events_path
            self.assertTrue(run_events_path.exists())

            run_events = _read_jsonl(run_events_path)
            phases = {row["phase"] for row in run_events}
            self.assertIn("run:start", phases)
            self.assertIn("run:finish", phases)

            state = store.load_state()
            task_id = state["tasks"][0]["id"]
            task_events_path = store.task_logs_dir / f"{task_id}.events.jsonl"
            task_memory_path = store.task_logs_dir / f"{task_id}.memory.jsonl"
            task_log_path = store.task_logs_dir / f"{task_id}.log"

            self.assertTrue(task_events_path.exists())
            self.assertTrue(task_memory_path.exists())
            self.assertTrue(task_log_path.exists())

            task_phases = {row["phase"] for row in _read_jsonl(task_events_path)}
            self.assertIn("task:start", task_phases)
            self.assertIn("task:completed", task_phases)

            memory_phases = {row["phase"] for row in _read_jsonl(task_memory_path)}
            self.assertIn("task:start", memory_phases)
            self.assertIn("after_cleanup", memory_phases)

    def test_structured_logging_records_failure_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                preprocessing_tasks=[_task("none")],
            )

            with patch(
                "pipeline.experiments.runner.run_training_task",
                side_effect=RuntimeError("synthetic runner failure"),
            ):
                snapshot = runner.run()

            self.assertEqual(snapshot["counts"]["failed"], 1)
            store = ExperimentStateStore(project_paths)
            state = store.load_state()
            task_id = state["tasks"][0]["id"]
            task_events = _read_jsonl(
                store.task_logs_dir / f"{task_id}.events.jsonl"
            )
            failed_events = [
                event for event in task_events if event.get("phase") == "task:failed"
            ]
            self.assertEqual(len(failed_events), 1)
            failed_event = failed_events[0]
            self.assertEqual(failed_event["error_type"], "RuntimeError")
            self.assertIn("synthetic runner failure", failed_event["error_message"])
            self.assertIn("traceback_summary", failed_event)
            self.assertIn("task_metadata", failed_event)
            self.assertIn("process_memory_mb", failed_event)


if __name__ == "__main__":
    unittest.main()
