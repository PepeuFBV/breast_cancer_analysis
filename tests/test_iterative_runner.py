from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

from pipeline.config import load_experiment_config
from pipeline.experiments import (
    ExperimentStateStore,
    IterativeExperimentRunner,
    IterativeRunOptions,
    launch_background_runner,
)
from pipeline.experiments.runner import (
    BACKGROUND_STDERR_LOG_ENV,
    BACKGROUND_STDOUT_LOG_ENV,
    MATERIALIZED_QUEUE_HARD_LIMIT,
    MAX_QUEUE_TASKS,
    REQUESTED_DEVICE_ENV,
    ExperimentGridCounts,
    build_experiment_record,
)
from pipeline.train.preprocessing import PreprocessingTask
from pipeline.train.runner import TrainingConfig, TrainingTask, artifact_paths_for_task
from pipeline.utils.paths import build_project_paths
from pipeline.utils.runtime_limits import CPU_THREAD_ENV_KEYS, CpuExecutionLimits


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
        param_id = "-".join([name, *[f"{key}-{value}" for key, value in resolved_params.items()]])
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


def _is_probe_runtime_command(command: object) -> bool:
    return "probe-runtime" in " ".join(str(item) for item in command)


def _probe_runtime_payload(env: dict[str, object]) -> dict[str, object]:
    requested_device = str(env.get(REQUESTED_DEVICE_ENV) or ("cpu" if env.get("CUDA_VISIBLE_DEVICES") == "-1" else "gpu"))
    effective_device = "cpu" if env.get("CUDA_VISIBLE_DEVICES") == "-1" else "gpu"
    cpu_thread_env = {key: env.get(key) for key in (*CPU_THREAD_ENV_KEYS, "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}
    return {
        "ok": True,
        "requested_device": requested_device,
        "effective_device": effective_device,
        "python_executable": sys.executable,
        "tensorflow_imported": True,
        "tensorflow_version": "test-tf",
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_devices": [] if effective_device == "cpu" else ["/physical_device:GPU:0"],
        "logical_gpu_devices": [] if effective_device == "cpu" else ["/device:GPU:0"],
        "tensorflow_visible_devices": [] if effective_device == "cpu" else ["/device:GPU:0"],
        "nvidia_smi_available": effective_device == "gpu",
        "nvidia_smi_command": [],
        "gpu_memory_summary": {"devices": []},
        "effective_cpu_thread_env": cpu_thread_env,
        "tensorflow_tiny_gpu_op": effective_device == "gpu",
        "tensorflow_tiny_gpu_op_device": ("/device:GPU:0" if effective_device == "gpu" else None),
        "gpu_used": effective_device == "gpu",
        "opencv_threads": None,
        "warnings": [],
        "errors": [],
    }


class IterativeRunnerTest(unittest.TestCase):
    def test_launch_background_runner_captures_stdout_and_stderr_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir = root / "artifacts" / "experiments" / "logs"
            script_path = root / "run_experiments.py"
            captured: dict[str, object] = {}

            def fake_popen(*args, **kwargs):
                captured.update(kwargs)
                captured["stdout"] = kwargs["stdout"]
                captured["stderr"] = kwargs["stderr"]
                return SimpleNamespace(pid=12345)

            with patch("pipeline.experiments.runner.subprocess.Popen", side_effect=fake_popen):
                launch = launch_background_runner(
                    script_path=script_path,
                    forwarded_args=["--limit", "2"],
                    cwd=root,
                    logs_dir=logs_dir,
                )

            self.assertEqual(launch.process.pid, 12345)
            self.assertTrue(logs_dir.exists())
            self.assertTrue(launch.stdout_path.exists())
            self.assertTrue(launch.stderr_path.exists())
            self.assertEqual(launch.stdout_path.suffixes[-2:], [".out", ".log"])
            self.assertEqual(launch.stderr_path.suffixes[-2:], [".err", ".log"])
            self.assertIsNot(captured["stdout"], subprocess.DEVNULL)
            self.assertIsNot(captured["stderr"], subprocess.DEVNULL)
            self.assertEqual(Path(str(getattr(captured["stdout"], "name"))), launch.stdout_path)
            self.assertEqual(Path(str(getattr(captured["stderr"], "name"))), launch.stderr_path)
            launch_env = cast(dict[str, str], captured["env"])
            self.assertEqual(launch_env[BACKGROUND_STDOUT_LOG_ENV], str(launch.stdout_path))
            self.assertEqual(launch_env[BACKGROUND_STDERR_LOG_ENV], str(launch.stderr_path))
            if os.name == "nt":
                self.assertNotIn("start_new_session", captured)
                self.assertGreater(int(captured.get("creationflags", 0)), 0)
            else:
                self.assertTrue(captured.get("start_new_session"))

    def test_isolated_runner_invokes_run_task_subprocess_for_each_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            tasks = [_task("none"), _task("none", {"variant": "second"})]
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.81)},
                preprocessing_tasks=tasks,
            )
            captured_commands: list[list[str]] = []
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                captured_commands.append([str(item) for item in command])
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="completed",
                    result_summary={"best_val_acc": 0.9},
                    duration_seconds=0.25,
                )
                return SimpleNamespace(returncode=0)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        run_task_command_base=(
                            "python",
                            "run_experiments.py",
                            "run-task",
                            "--artifacts-dir",
                            str(project_paths.artifacts_dir),
                        ),
                    )
                )

            self.assertEqual(len(captured_commands), 2)
            self.assertTrue(all("--task-id" in command for command in captured_commands))
            self.assertTrue(all("--artifacts-dir" in command for command in captured_commands))
            self.assertEqual(snapshot["counts"]["completed"], 2)
            self.assertEqual(snapshot["counts"]["failed"], 0)

    def test_isolated_runner_continues_after_failed_child_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            tasks = [_task(f"none_{index}") for index in range(3)]
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.83)},
                preprocessing_tasks=tasks,
            )
            calls = {"count": 0}
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                calls["count"] += 1
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                if calls["count"] == 1:
                    runner.store.update_task_status(
                        task_id,
                        status="failed",
                        error_summary="synthetic child failure",
                        duration_seconds=0.2,
                    )
                    return SimpleNamespace(returncode=1)
                runner.store.update_task_status(
                    task_id,
                    status="completed",
                    result_summary={"best_val_acc": 0.88},
                    duration_seconds=0.2,
                )
                return SimpleNamespace(returncode=0)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                    )
                )

            self.assertEqual(calls["count"], 3)
            self.assertEqual(snapshot["counts"]["failed"], 1)
            self.assertEqual(snapshot["counts"]["completed"], 2)
            state = runner.store.load_state()
            statuses = [task["status"] for task in state["tasks"]]
            self.assertEqual(statuses[0], "failed")
            self.assertEqual(statuses[1:], ["completed", "completed"])

    def test_isolated_runner_stops_before_next_task_when_stop_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            tasks = [_task(f"none_{index}") for index in range(3)]
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.85)},
                preprocessing_tasks=tasks,
            )
            calls = {"count": 0}
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                calls["count"] += 1
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="completed",
                    result_summary={"best_val_acc": 0.91},
                    duration_seconds=0.1,
                )
                runner.store.request_stop(reason="test-stop")
                return SimpleNamespace(returncode=0)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                    )
                )

            self.assertEqual(calls["count"], 1)
            self.assertEqual(snapshot["counts"]["completed"], 1)
            self.assertEqual(snapshot["counts"]["pending"], 2)
            self.assertTrue(snapshot["stop_requested"])

    def test_isolated_runner_records_child_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=[_task("none")],
            )
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                return SimpleNamespace(returncode=7)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                    )
                )

            self.assertEqual(snapshot["counts"]["failed"], 1)
            state = runner.store.load_state()
            self.assertIn("exited with code 7", state["tasks"][0]["error_summary"])
            run_events = _read_jsonl(runner.store.run_events_path)
            subprocess_events = [row for row in run_events if row.get("phase") == "task:subprocess_exit"]
            self.assertEqual(len(subprocess_events), 1)
            self.assertEqual(subprocess_events[0]["subprocess_exit_code"], 7)

    @pytest.mark.gpu
    def test_adaptive_policy_retries_gpu_then_falls_back_to_cpu_and_recovers_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            tasks = [_task("none"), _task("none", {"variant": "second"})]
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=tasks,
            )
            outcomes = [
                ("gpu", 1, "failed", "ResourceExhaustedError: OOM"),
                ("gpu", 1, "failed", "CUDA_ERROR_OUT_OF_MEMORY"),
                ("cpu", 0, "completed", None),
                ("gpu", 0, "completed", None),
            ]
            seen_devices: list[str] = []
            original_run = subprocess.run
            call_count = {"value": 0}

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                outcome = outcomes[call_count["value"]]
                call_count["value"] += 1
                expected_device, returncode, status, error_summary = outcome
                env = kwargs.get("env", {})
                observed_device = "cpu" if env.get("CUDA_VISIBLE_DEVICES") == "-1" else "gpu"
                seen_devices.append(observed_device)
                self.assertEqual(observed_device, expected_device)
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                if status == "completed":
                    runner.store.update_task_status(
                        task_id,
                        status="completed",
                        result_summary={"best_val_acc": 0.9},
                        duration_seconds=0.2,
                    )
                else:
                    runner.store.update_task_status(
                        task_id,
                        status="failed",
                        error_summary=error_summary,
                        duration_seconds=0.2,
                    )
                return SimpleNamespace(returncode=returncode)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        device_policy="adaptive",
                        gpu_retries=1,
                        cpu_retries=1,
                        cooldown_after_oom_seconds=0,
                        gpu_recovery_cooldown_seconds=0,
                        max_task_attempts=4,
                    )
                )

            self.assertEqual(seen_devices, ["gpu", "gpu", "cpu", "gpu"])
            self.assertEqual(snapshot["counts"]["completed"], 2)
            self.assertEqual(snapshot["gpu_oom_count"], 2)
            self.assertEqual(snapshot["cpu_fallback_successes"], 1)
            self.assertEqual(snapshot["preferred_device"], "gpu")

            state = runner.store.load_state()
            first_task = state["tasks"][0]
            second_task = state["tasks"][1]
            self.assertEqual(first_task["status"], "completed")
            self.assertEqual(first_task["final_device"], "cpu")
            self.assertEqual(first_task["gpu_attempts"], 2)
            self.assertEqual(first_task["cpu_attempts"], 1)
            self.assertEqual(len(first_task["attempt_history"]), 3)
            self.assertEqual(second_task["final_device"], "gpu")

    @pytest.mark.gpu
    def test_gpu_attempt_preserves_visible_devices_and_records_probe_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=[_task("none")],
            )
            captured_env: dict[str, str] = {}
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                captured_env.update({str(key): str(value) for key, value in kwargs.get("env", {}).items()})
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="completed",
                    result_summary={"best_val_acc": 0.93},
                    duration_seconds=0.1,
                )
                return SimpleNamespace(returncode=0)

            with (
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2"}, clear=False),
                patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run),
            ):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        device_policy="gpu-only",
                        max_task_attempts=1,
                    )
                )

            self.assertEqual(snapshot["counts"]["completed"], 1)
            self.assertEqual(captured_env["CUDA_VISIBLE_DEVICES"], "0,2")
            state = runner.store.load_state()
            attempt = state["tasks"][0]["attempt_history"][0]
            self.assertEqual(attempt["requested_device"], "gpu")
            self.assertEqual(attempt["effective_device"], "gpu")
            self.assertEqual(attempt["cuda_visible_devices"], "0,2")
            self.assertEqual(attempt["tensorflow_visible_devices"], ["/device:GPU:0"])
            self.assertTrue(attempt["gpu_used"])

    def test_cpu_attempt_sets_thread_limit_env_vars_and_attempt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                cpu_execution_limits=CpuExecutionLimits(
                    max_threads=2,
                    opencv_threads=1,
                    inter_op_threads=1,
                    intra_op_threads=2,
                ),
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=[_task("none")],
            )
            captured_env: dict[str, str] = {}
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                captured_env.update({str(key): str(value) for key, value in kwargs.get("env", {}).items()})
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="completed",
                    result_summary={"best_val_acc": 0.92},
                    duration_seconds=0.1,
                )
                return SimpleNamespace(returncode=0)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        device_policy="cpu-only",
                        max_task_attempts=1,
                    )
                )

            self.assertEqual(snapshot["counts"]["completed"], 1)
            self.assertEqual(captured_env["CUDA_VISIBLE_DEVICES"], "-1")
            self.assertEqual(captured_env["OMP_NUM_THREADS"], "2")
            self.assertEqual(captured_env["OPENBLAS_NUM_THREADS"], "2")
            self.assertEqual(captured_env["MKL_NUM_THREADS"], "2")
            self.assertEqual(captured_env["NUMEXPR_NUM_THREADS"], "2")
            self.assertEqual(captured_env["VECLIB_MAXIMUM_THREADS"], "2")
            self.assertEqual(captured_env["TF_NUM_INTRAOP_THREADS"], "2")
            self.assertEqual(captured_env["TF_NUM_INTEROP_THREADS"], "1")
            state = runner.store.load_state()
            attempt = state["tasks"][0]["attempt_history"][0]
            self.assertEqual(attempt["requested_device"], "cpu")
            self.assertEqual(attempt["effective_device"], "cpu")
            self.assertEqual(attempt["effective_cpu_thread_env"]["OMP_NUM_THREADS"], "2")
            self.assertFalse(attempt["gpu_used"])

    @pytest.mark.gpu
    def test_adaptive_gpu_probe_failure_falls_back_to_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=[_task("none")],
            )
            task_devices: list[str] = []
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                env = kwargs.get("env", {})
                if _is_probe_runtime_command(command):
                    payload = _probe_runtime_payload(env)
                    if env.get("CUDA_VISIBLE_DEVICES") != "-1":
                        payload.update(
                            {
                                "ok": False,
                                "effective_device": "cpu",
                                "physical_gpu_devices": [],
                                "logical_gpu_devices": [],
                                "tensorflow_visible_devices": [],
                                "tensorflow_tiny_gpu_op": False,
                                "tensorflow_tiny_gpu_op_device": None,
                                "gpu_used": False,
                                "errors": ["No TensorFlow GPU devices are visible."],
                            }
                        )
                        return SimpleNamespace(returncode=1, stdout=json.dumps(payload), stderr="")
                    return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                observed_device = "cpu" if env.get("CUDA_VISIBLE_DEVICES") == "-1" else "gpu"
                task_devices.append(observed_device)
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="completed",
                    result_summary={"best_val_acc": 0.91},
                    duration_seconds=0.1,
                )
                return SimpleNamespace(returncode=0)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        device_policy="adaptive",
                        gpu_retries=0,
                        cpu_retries=1,
                        gpu_recovery_cooldown_seconds=0,
                        max_task_attempts=2,
                    )
                )

            self.assertEqual(snapshot["counts"]["completed"], 1)
            self.assertEqual(task_devices, ["cpu"])
            state = runner.store.load_state()
            task = state["tasks"][0]
            self.assertEqual(task["final_device"], "cpu")
            self.assertEqual(len(task["attempt_history"]), 2)
            self.assertEqual(task["attempt_history"][0]["failure_kind"], "gpu_unavailable")
            self.assertEqual(task["attempt_history"][1]["requested_device"], "cpu")
            run_events = _read_jsonl(runner.store.run_events_path)
            phases = {row.get("event") for row in run_events}
            self.assertIn("gpu_probe_failed", phases)
            self.assertIn("cpu_fallback_scheduled", phases)

    @pytest.mark.gpu
    def test_adaptive_cooldown_is_not_extended_by_cpu_success_without_gpu_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=[
                    _task("none"),
                    _task("none", {"variant": "second"}),
                    _task("none", {"variant": "third"}),
                ],
            )
            gpu_probe_calls = {"count": 0}
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                env = kwargs.get("env", {})
                if _is_probe_runtime_command(command):
                    payload = _probe_runtime_payload(env)
                    if env.get("CUDA_VISIBLE_DEVICES") != "-1":
                        gpu_probe_calls["count"] += 1
                        payload.update(
                            {
                                "ok": False,
                                "effective_device": "cpu",
                                "physical_gpu_devices": [],
                                "logical_gpu_devices": [],
                                "tensorflow_visible_devices": [],
                                "tensorflow_tiny_gpu_op": False,
                                "tensorflow_tiny_gpu_op_device": None,
                                "gpu_used": False,
                                "errors": ["No TensorFlow GPU devices are visible."],
                            }
                        )
                        return SimpleNamespace(returncode=1, stdout=json.dumps(payload), stderr="")
                    return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                if env.get("CUDA_VISIBLE_DEVICES") == "-1":
                    time.sleep(0.65)
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="completed",
                    result_summary={"best_val_acc": 0.91},
                    duration_seconds=0.1,
                )
                return SimpleNamespace(returncode=0)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        device_policy="adaptive",
                        gpu_retries=0,
                        cpu_retries=0,
                        cooldown_after_oom_seconds=0,
                        gpu_recovery_cooldown_seconds=1,
                        max_task_attempts=2,
                    )
                )

            self.assertEqual(snapshot["counts"]["completed"], 3)
            self.assertEqual(gpu_probe_calls["count"], 2)

    def test_cpu_limits_are_applied_for_direct_cpu_task_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            original_opencv_threads = cv2.getNumThreads()
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                cpu_execution_limits=CpuExecutionLimits(
                    max_threads=2,
                    opencv_threads=1,
                    inter_op_threads=1,
                    intra_op_threads=2,
                ),
                preprocessing_tasks=[_task("none")],
            )
            record, _training_task = runner.build_queue()[0]
            observed: dict[str, object] = {}

            def _failing_run_training_task(*args, **kwargs):
                observed["omp"] = os.environ.get("OMP_NUM_THREADS")
                observed["openblas"] = os.environ.get("OPENBLAS_NUM_THREADS")
                observed["mkl"] = os.environ.get("MKL_NUM_THREADS")
                observed["numexpr"] = os.environ.get("NUMEXPR_NUM_THREADS")
                observed["veclib"] = os.environ.get("VECLIB_MAXIMUM_THREADS")
                observed["tf_intra"] = os.environ.get("TF_NUM_INTRAOP_THREADS")
                observed["tf_inter"] = os.environ.get("TF_NUM_INTEROP_THREADS")
                observed["opencv_threads"] = cv2.getNumThreads()
                raise RuntimeError("synthetic cpu execution stop")

            try:
                with (
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1", REQUESTED_DEVICE_ENV: "cpu"}, clear=False),
                    patch("pipeline.experiments.runner.run_training_task", side_effect=_failing_run_training_task),
                ):
                    result = runner.run_one_task(record["id"], task_cooldown_seconds=0)
            finally:
                cv2.setNumThreads(original_opencv_threads)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(observed["omp"], "2")
            self.assertEqual(observed["openblas"], "2")
            self.assertEqual(observed["mkl"], "2")
            self.assertEqual(observed["numexpr"], "2")
            self.assertEqual(observed["veclib"], "2")
            self.assertEqual(observed["tf_intra"], "2")
            self.assertEqual(observed["tf_inter"], "1")
            self.assertEqual(observed["opencv_threads"], 1)
            run_events = _read_jsonl(runner.store.run_events_path)
            cpu_limit_events = [row for row in run_events if row.get("event") == "cpu_limits_applied"]
            self.assertGreaterEqual(len(cpu_limit_events), 1)

    def test_cpu_only_policy_never_uses_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            tasks = [_task("none"), _task("none", {"variant": "second"})]
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=tasks,
            )
            seen_devices: list[str] = []
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                env = kwargs.get("env", {})
                observed_device = "cpu" if env.get("CUDA_VISIBLE_DEVICES") == "-1" else "gpu"
                seen_devices.append(observed_device)
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="completed",
                    result_summary={"best_val_acc": 0.9},
                    duration_seconds=0.1,
                )
                return SimpleNamespace(returncode=0)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        device_policy="cpu-only",
                    )
                )

            self.assertEqual(snapshot["counts"]["completed"], 2)
            self.assertEqual(seen_devices, ["cpu", "cpu"])

    @pytest.mark.gpu
    def test_gpu_only_policy_never_falls_back_to_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=[_task("none")],
            )
            seen_devices: list[str] = []
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                env = kwargs.get("env", {})
                observed_device = "cpu" if env.get("CUDA_VISIBLE_DEVICES") == "-1" else "gpu"
                seen_devices.append(observed_device)
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="failed",
                    error_summary="ResourceExhaustedError: OOM",
                    duration_seconds=0.1,
                )
                return SimpleNamespace(returncode=1)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        device_policy="gpu-only",
                        gpu_retries=1,
                        max_task_attempts=2,
                    )
                )

            self.assertEqual(snapshot["counts"]["failed"], 1)
            self.assertEqual(seen_devices, ["gpu", "gpu"])

    @pytest.mark.gpu
    def test_gpu_only_policy_fails_fast_when_gpu_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=[_task("none")],
            )

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    payload = _probe_runtime_payload(kwargs.get("env", {}))
                    payload.update(
                        {
                            "ok": False,
                            "effective_device": "cpu",
                            "physical_gpu_devices": [],
                            "logical_gpu_devices": [],
                            "tensorflow_visible_devices": [],
                            "tensorflow_tiny_gpu_op": False,
                            "tensorflow_tiny_gpu_op_device": None,
                            "gpu_used": False,
                            "errors": ["No TensorFlow GPU devices are visible."],
                        }
                    )
                    return SimpleNamespace(returncode=1, stdout=json.dumps(payload), stderr="")
                raise AssertionError("run-task subprocess should not start when gpu-only validation fails.")

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                with self.assertRaises(RuntimeError) as context:
                    runner.run(
                        IterativeRunOptions(
                            isolate_tasks=True,
                            task_cooldown_seconds=0,
                            device_policy="gpu-only",
                        )
                    )

            self.assertIn("gpu-only", str(context.exception))

    def test_fail_fast_on_oom_stops_retries_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=[_task("none")],
            )
            seen_devices: list[str] = []
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                env = kwargs.get("env", {})
                observed_device = "cpu" if env.get("CUDA_VISIBLE_DEVICES") == "-1" else "gpu"
                seen_devices.append(observed_device)
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="failed",
                    error_summary="ResourceExhaustedError: OOM",
                    duration_seconds=0.1,
                )
                return SimpleNamespace(returncode=1)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        device_policy="gpu-first",
                        gpu_retries=3,
                        cpu_retries=3,
                        fail_fast_on_oom=True,
                    )
                )

            self.assertEqual(snapshot["counts"]["failed"], 1)
            self.assertEqual(seen_devices, ["gpu"])
            state = runner.store.load_state()
            self.assertEqual(len(state["tasks"][0]["attempt_history"]), 1)

    def test_consecutive_oom_limit_stops_new_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            tasks = [_task("none"), _task("none", {"variant": "second"}), _task("none", {"variant": "third"})]
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.86)},
                preprocessing_tasks=tasks,
            )
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="failed",
                    error_summary="ResourceExhaustedError: OOM",
                    duration_seconds=0.1,
                )
                return SimpleNamespace(returncode=1)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        device_policy="gpu-only",
                        gpu_retries=0,
                        max_consecutive_oom=2,
                        max_task_attempts=1,
                    )
                )

            self.assertEqual(snapshot["counts"]["failed"], 2)
            self.assertEqual(snapshot["counts"]["pending"], 1)
            self.assertTrue(snapshot["stop_requested"])
            self.assertEqual(snapshot["consecutive_final_oom_failures"], 2)
            self.assertIn("max_consecutive_oom=2", str(snapshot.get("oom_policy_stop")))

            run_events = _read_jsonl(runner.store.run_events_path)
            attempt_finished = [row for row in run_events if row.get("event") == "task_attempt_finished"]
            self.assertGreaterEqual(len(attempt_finished), 2)
            for event in attempt_finished:
                self.assertIn("device", event)
                self.assertIn("exit_code", event)
                self.assertIn("failure_kind", event)

    def test_isolated_runner_marks_task_failed_on_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.89)},
                preprocessing_tasks=[_task("none")],
            )
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout", 0))

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        task_timeout_seconds=1.0,
                    )
                )

            self.assertEqual(snapshot["counts"]["failed"], 1)
            state = runner.store.load_state()
            self.assertIn("timed out", state["tasks"][0]["error_summary"])
            run_events = _read_jsonl(runner.store.run_events_path)
            timeout_events = [row for row in run_events if row.get("phase") == "task:timeout"]
            self.assertEqual(len(timeout_events), 1)

    def test_run_forces_isolation_even_when_disabled_in_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.87)},
                preprocessing_tasks=[_task("none")],
            )
            original_run = subprocess.run

            def _fake_subprocess_run(command, **kwargs):
                if _is_probe_runtime_command(command):
                    return SimpleNamespace(returncode=0, stdout=json.dumps(_probe_runtime_payload(kwargs.get("env", {}))), stderr="")
                if "--task-id" not in command:
                    return original_run(command, **kwargs)
                task_id = command[command.index("--task-id") + 1]
                runner.store.update_task_status(task_id, status="running")
                runner.store.update_task_status(
                    task_id,
                    status="completed",
                    result_summary={"best_val_acc": 0.9},
                    duration_seconds=0.1,
                )
                return SimpleNamespace(returncode=0)

            with patch("pipeline.experiments.runner.subprocess.run", side_effect=_fake_subprocess_run):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=False,
                        task_cooldown_seconds=0,
                    )
                )
            self.assertEqual(snapshot["counts"]["completed"], 1)
            run_events = _read_jsonl(runner.store.run_events_path)
            subprocess_start_events = [row for row in run_events if row.get("phase") == "task:subprocess_start"]
            self.assertEqual(len(subprocess_start_events), 1)

    def test_write_pid_record_preserves_background_log_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            project_paths = build_project_paths(root / "raw-data", root / "artifacts")
            store = ExperimentStateStore(project_paths)
            store.ensure_dirs()
            stdout_path = project_paths.experiment_logs_dir / "background-runner.out.log"
            stderr_path = project_paths.experiment_logs_dir / "background-runner.err.log"

            store.write_pid_record(
                config_path=Path("configs/experiment.default.json"),
                command=["python", "run_experiments.py", "run"],
                pid=4444,
                stdout_log_path=stdout_path,
                stderr_log_path=stderr_path,
            )
            store.write_pid_record(
                config_path=Path("configs/experiment.default.json"),
                command=["python", "run_experiments.py", "run"],
                pid=4444,
            )

            record = store.read_pid_record()
            assert record is not None
            self.assertEqual(record["pid"], 4444)
            self.assertEqual(record["stdout_log_path"], str(stdout_path))
            self.assertEqual(record["stderr_log_path"], str(stderr_path))

    def test_reset_refuses_active_run_without_kill_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            project_paths = build_project_paths(root / "raw-data", root / "artifacts")
            store = ExperimentStateStore(project_paths)
            store.ensure_dirs()
            store.write_pid_record(
                config_path=Path("configs/experiment.default.json"),
                command=["python", "run_experiments.py", "run"],
                pid=4444,
            )

            with patch("pipeline.experiments.runner._is_process_alive", return_value=True):
                with self.assertRaises(RuntimeError) as context:
                    store.reset(purge_results=True)

            self.assertIn("reset --kill-active", str(context.exception))

    def test_reset_kills_active_run_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            project_paths = build_project_paths(root / "raw-data", root / "artifacts")
            store = ExperimentStateStore(project_paths)
            store.ensure_dirs()
            store.write_pid_record(
                config_path=Path("configs/experiment.default.json"),
                command=["python", "run_experiments.py", "run"],
                pid=4444,
            )

            with (
                patch("pipeline.experiments.runner._is_process_alive", return_value=True),
                patch.object(store, "terminate_active_run", return_value=4444) as terminate_active_run,
            ):
                store.reset(purge_results=True, kill_active=True)

            terminate_active_run.assert_called_once_with(force=True)

    def test_build_experiment_id_does_not_resolve_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
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
            self.assertEqual(len(list(project_paths.experiment_task_dir.glob("*.json"))), 12)

    def test_status_reconciles_stale_running_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
                preprocessing_tasks=[_task("none")],
            )
            record, _ = runner.build_queue()[0]
            store = ExperimentStateStore(project_paths)
            store.ensure_dirs()
            store.sync_queue([record], config_path=Path("configs/experiment.default.json"))
            store.update_task_status(record["id"], status="running")

            snapshot = store.summarize()
            state = store.load_state()

            self.assertEqual(snapshot["counts"]["running"], 0)
            self.assertEqual(snapshot["counts"]["stopped"], 1)
            self.assertEqual(state["tasks"][0]["status"], "stopped")

    def test_run_allows_active_pid_record_when_pid_is_current_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
                preprocessing_tasks=[_task("none")],
            )
            store = ExperimentStateStore(project_paths)
            store.ensure_dirs()
            store.write_pid_record(
                config_path=Path("configs/experiment.default.json"),
                command=["python", "run_experiments.py", "run"],
                pid=os.getpid(),
            )

            snapshot = runner.run()

            self.assertEqual(snapshot["counts"]["completed"], 1)

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
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
            )

            with self.assertRaises(ValueError) as context:
                runner.build_queue()

            self.assertIn("safety limit", str(context.exception))

    def test_runner_allows_large_queue_with_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
            )

            with (
                patch(
                    "pipeline.experiments.runner.count_preprocessing_tasks",
                    return_value=MAX_QUEUE_TASKS + 1,
                ),
                patch(
                    "pipeline.experiments.runner.build_training_tasks",
                    return_value=[],
                ),
            ):
                queue = runner.build_queue_with_options(max_queue_tasks=MAX_QUEUE_TASKS + 10)

            self.assertEqual(queue, [])

    def test_run_streams_huge_queue_without_materializing_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
            )
            record = build_experiment_record(
                TrainingTask(
                    model_name="custom cnn",
                    preprocessing_task=_task("none"),
                    augmentations_per_image=1,
                ),
                config,
            )

            def _fake_streamed_task_run(*, record, options, runtime_state):
                runner.store.register_task_record(record, config_path=runner.config_path)
                runner.store.update_task_status(
                    record["id"],
                    status="completed",
                    result_summary={"status": "ok"},
                    duration_seconds=0.1,
                )
                return {"status": "completed"}

            with (
                patch.object(
                    runner,
                    "estimate_grid_counts",
                    return_value=ExperimentGridCounts(
                        preprocessing_count=1,
                        model_count=1,
                        augmentation_count=1,
                        total_experiments=MATERIALIZED_QUEUE_HARD_LIMIT + 1,
                        total_fits=MATERIALIZED_QUEUE_HARD_LIMIT + 1,
                        folds=1,
                    ),
                ),
                patch.object(
                    runner,
                    "_build_queue_and_sync_state",
                    side_effect=AssertionError("streamed run should not materialize the queue"),
                ),
                patch.object(runner, "_iter_queue_entries", return_value=iter([(record, None)])),
                patch.object(
                    runner,
                    "_run_task_with_isolated_device_policy",
                    side_effect=_fake_streamed_task_run,
                ),
            ):
                snapshot = runner.run(
                    IterativeRunOptions(
                        isolate_tasks=True,
                        task_cooldown_seconds=0,
                        max_queue_tasks=None,
                        limit=1,
                    )
                )

            self.assertTrue(snapshot["stream_queue_mode"])
            self.assertEqual(snapshot["counts"]["completed"], 1)
            self.assertEqual(snapshot["counts"]["pending"], MATERIALIZED_QUEUE_HARD_LIMIT)

    def test_build_queue_can_export_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
                preprocessing_tasks=[_task("none"), _task("none", {"variant": "second"})],
            )

            queue_export_path = root / "queue-export.jsonl"
            queue = runner.build_queue_with_options(queue_export_path=queue_export_path)

            self.assertEqual(len(queue), 2)
            lines = queue_export_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            metadata = json.loads(lines[0])
            first_task = json.loads(lines[1])
            self.assertEqual(metadata["kind"], "metadata")
            self.assertEqual(metadata["task_count"], 2)
            self.assertEqual(first_task["kind"], "task")
            self.assertIn("id", first_task)

    def test_keyboard_interrupt_marks_current_task_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
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
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
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
            task_events = _read_jsonl(store.task_logs_dir / f"{task_id}.events.jsonl")
            failed_events = [event for event in task_events if event.get("phase") == "task:failed"]
            self.assertEqual(len(failed_events), 1)
            failed_event = failed_events[0]
            self.assertEqual(failed_event["error_type"], "RuntimeError")
            self.assertIn("synthetic runner failure", failed_event["error_message"])
            self.assertIn("traceback_summary", failed_event)
            self.assertIn("task_metadata", failed_event)
            self.assertIn("process_memory_mb", failed_event)

    def test_run_one_task_saves_artifacts_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.9)},
                preprocessing_tasks=[_task("none")],
            )
            record, training_task = runner.build_queue()[0]

            result = runner.run_one_task(record["id"])

            self.assertEqual(result["status"], "completed")
            self.assertFalse(result.get("skipped", False))
            history_path, predictions_path = artifact_paths_for_task(config, training_task)
            self.assertTrue(history_path.exists())
            self.assertTrue(predictions_path.exists())

            store = ExperimentStateStore(project_paths)
            state = store.load_state()
            self.assertEqual(state["tasks"][0]["status"], "completed")
            self.assertEqual(state["tasks"][0]["attempts"], 1)

    def test_run_one_task_fails_for_missing_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
                preprocessing_tasks=[_task("none")],
            )

            with self.assertRaises(ValueError) as context:
                runner.run_one_task("exp-missing")

            self.assertIn("was not found", str(context.exception))

    def test_run_one_task_marks_failed_tasks_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
                preprocessing_tasks=[_task("none")],
            )
            record, _ = runner.build_queue()[0]

            with patch(
                "pipeline.experiments.runner.run_training_task",
                side_effect=RuntimeError("single-task failure"),
            ):
                result = runner.run_one_task(record["id"])

            self.assertEqual(result["status"], "failed")
            store = ExperimentStateStore(project_paths)
            state = store.load_state()
            self.assertEqual(state["tasks"][0]["status"], "failed")
            self.assertIn("single-task failure", state["tasks"][0]["error_summary"])

    def test_run_one_task_does_not_corrupt_existing_queue_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            tasks = [_task("none"), _task("none", {"variant": "second"})]
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.82)},
                preprocessing_tasks=tasks,
            )
            queue_entries = runner.build_queue()
            first_id = queue_entries[0][0]["id"]
            second_id = queue_entries[1][0]["id"]

            store = ExperimentStateStore(project_paths)
            store.ensure_dirs()
            store.sync_queue([record for record, _ in queue_entries], config_path=Path("configs/experiment.default.json"))
            store.update_task_status(second_id, status="failed", error_summary="preexisting failure")

            result = runner.run_one_task(first_id)

            self.assertEqual(result["status"], "completed")
            state = store.load_state()
            status_by_id = {task["id"]: task for task in state["tasks"]}
            self.assertEqual(status_by_id[first_id]["status"], "completed")
            self.assertEqual(status_by_id[second_id]["status"], "failed")
            self.assertEqual(status_by_id[second_id]["error_summary"], "preexisting failure")

    def test_run_one_task_respects_artifact_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, project_paths = _build_training_config(root)
            runner = IterativeExperimentRunner(
                config_path=Path("configs/experiment.default.json"),
                project_paths=project_paths,
                training_config=config,
                model_builders={"custom cnn": lambda *args, **kwargs: _CountingModel(0.8)},
                preprocessing_tasks=[_task("none")],
            )
            record, training_task = runner.build_queue()[0]
            history_path, predictions_path = artifact_paths_for_task(config, training_task)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "best_val_acc": 0.99,
                        "best_epoch": 1,
                        "preproc_id": training_task.preproc_id,
                        "model_name": training_task.model_name,
                        "param_id": training_task.param_id,
                    }
                ]
            ).to_csv(history_path, index=False)
            pd.DataFrame([{"y_true": 0, "y_pred": 0}]).to_csv(predictions_path, index=False)

            with patch(
                "pipeline.experiments.runner.run_training_task",
                side_effect=AssertionError("run_training_task should not be called for reconciled tasks"),
            ):
                result = runner.run_one_task(record["id"])

            self.assertEqual(result["status"], "completed")
            self.assertTrue(result.get("skipped"))
            self.assertEqual(result.get("skip_reason"), "reconciled_from_existing_artifacts")

            store = ExperimentStateStore(project_paths)
            state = store.load_state()
            self.assertEqual(state["tasks"][0]["status"], "completed")
            self.assertGreaterEqual(int(state["tasks"][0]["attempts"]), 1)


if __name__ == "__main__":
    unittest.main()
