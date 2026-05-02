from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _import_run_experiments_module():
    os.environ["BREAST_CANCER_ANALYSIS_GPU_ENV_BOOTSTRAPPED"] = "1"
    module = sys.modules.get("run_experiments")
    if module is not None:
        return module
    return importlib.import_module("run_experiments")


def test_run_command_returns_130_on_keyboard_interrupt_before_state(
    capsys,
    monkeypatch,
) -> None:
    module = _import_run_experiments_module()
    fake_runner = SimpleNamespace(
        store=SimpleNamespace(
            summarize=lambda: {
                "total": 0,
                "counts": {
                    "pending": 0,
                    "running": 0,
                    "completed": 0,
                    "failed": 0,
                    "stopped": 0,
                },
                "overall_status": "idle",
                "current_task": None,
                "state_path": "/tmp/missing-runner-state.json",
                "summary_path": "/tmp/missing-summary.csv",
                "history_dir": "/tmp/history",
                "predictions_dir": "/tmp/predictions",
                "log_path": "/tmp/runner.log",
                "active_pid": None,
                "stop_requested": False,
            }
        ),
    )

    def _interrupting_run(_options):
        raise KeyboardInterrupt("manual stop")

    fake_runner.run = _interrupting_run

    monkeypatch.setattr(module, "_build_runner", lambda args: fake_runner)

    exit_code = module.main(["run"])

    output = capsys.readouterr().out
    assert exit_code == 130
    assert "Run interrupted before any experiment state was created." in output


def test_print_status_snapshot_shows_background_logs_when_present(capsys) -> None:
    module = _import_run_experiments_module()
    snapshot = {
        "overall_status": "running",
        "total": 2,
        "counts": {
            "pending": 1,
            "running": 0,
            "completed": 1,
            "failed": 0,
            "stopped": 0,
        },
        "current_task": None,
        "state_path": "/tmp/state.json",
        "summary_path": "/tmp/summary.csv",
        "history_dir": "/tmp/history",
        "predictions_dir": "/tmp/predictions",
        "log_path": "/tmp/runner.log",
        "active_pid": 123,
        "background_stdout_log_path": "/tmp/background.out.log",
        "background_stderr_log_path": "/tmp/background.err.log",
        "device_policy": "adaptive",
        "preferred_device": "gpu",
        "gpu_health": "healthy",
        "last_gpu_oom_task_id": "exp-oom",
        "gpu_oom_count": 2,
        "cpu_fallback_successes": 1,
        "consecutive_final_oom_failures": 0,
        "last_successful_device": "cpu",
    }

    module._print_status_snapshot(snapshot)
    output = capsys.readouterr().out
    assert "Background stdout log: /tmp/background.out.log" in output
    assert "Background stderr log: /tmp/background.err.log" in output
    assert "Device policy: adaptive" in output
    assert "Current preferred device: gpu" in output
    assert "GPU OOM count: 2" in output


def test_print_status_snapshot_works_without_background_logs(capsys) -> None:
    module = _import_run_experiments_module()
    snapshot = {
        "overall_status": "idle",
        "total": 0,
        "counts": {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "stopped": 0,
        },
        "current_task": None,
        "state_path": "/tmp/state.json",
        "summary_path": "/tmp/summary.csv",
        "history_dir": "/tmp/history",
        "predictions_dir": "/tmp/predictions",
        "log_path": "/tmp/runner.log",
        "active_pid": None,
    }

    module._print_status_snapshot(snapshot)
    output = capsys.readouterr().out
    assert "Overall status: idle" in output


def test_run_task_command_returns_zero_when_completed(
    capsys,
    monkeypatch,
) -> None:
    module = _import_run_experiments_module()
    fake_runner = SimpleNamespace(
        config_path="configs/experiment.default.json",
        project_paths=SimpleNamespace(),
        training_config=SimpleNamespace(),
    )
    monkeypatch.setattr(module, "_build_runner", lambda args: fake_runner)
    monkeypatch.setattr(
        module,
        "run_one_experiment_task",
        lambda **kwargs: {"task_id": kwargs["task_id"], "status": "completed"},
    )

    exit_code = module.main(["run-task", "--task-id", "exp-abc123"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Task exp-abc123 completed successfully." in output


def test_run_task_command_shows_clear_error_for_missing_task(
    capsys,
    monkeypatch,
) -> None:
    module = _import_run_experiments_module()
    fake_runner = SimpleNamespace(
        config_path="configs/experiment.default.json",
        project_paths=SimpleNamespace(),
        training_config=SimpleNamespace(),
    )
    monkeypatch.setattr(module, "_build_runner", lambda args: fake_runner)

    def _missing_task(**kwargs):
        raise ValueError("Task id 'exp-missing' was not found.")

    monkeypatch.setattr(module, "run_one_experiment_task", _missing_task)

    exit_code = module.main(["run-task", "--task-id", "exp-missing"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Task id 'exp-missing' was not found." in output


def test_run_command_forwards_isolation_options_and_artifacts_dir(
    capsys,
    monkeypatch,
) -> None:
    module = _import_run_experiments_module()
    captured = {"options": None}

    def _fake_run(options):
        captured["options"] = options
        return {
            "overall_status": "idle",
            "total": 0,
            "counts": {
                "pending": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "stopped": 0,
            },
            "current_task": None,
            "state_path": "/tmp/state.json",
            "summary_path": "/tmp/summary.csv",
            "history_dir": "/tmp/history",
            "predictions_dir": "/tmp/predictions",
            "log_path": "/tmp/runner.log",
            "active_pid": None,
        }

    fake_runner = SimpleNamespace(
        run=_fake_run,
        store=SimpleNamespace(summarize=lambda: {}),
    )

    monkeypatch.setattr(module, "_build_runner", lambda args: fake_runner)
    monkeypatch.setattr(
        module,
        "load_experiment_config",
        lambda _path: SimpleNamespace(
            source_path=Path("configs/experiment.default.json"),
            runner=SimpleNamespace(
                isolate_tasks=False,
                task_cooldown_seconds=2.0,
                task_timeout_seconds=None,
            ),
        ),
    )

    exit_code = module.main(
        [
            "run",
            "--isolate-tasks",
            "--task-cooldown-seconds",
            "4",
            "--task-timeout-seconds",
            "11",
            "--max-queue-tasks",
            "123456",
            "--queue-export-path",
            "/tmp/queue-export.jsonl",
            "--artifacts-dir",
            "/tmp/isolation-artifacts",
            "--models",
            "custom cnn",
            "--preprocessing",
            "none",
            "--no-combined-preprocessing",
            "--no-run-skip",
        ]
    )

    assert exit_code == 0
    options = captured["options"]
    assert options is not None
    assert options.isolate_tasks is True
    assert options.task_cooldown_seconds == 4.0
    assert options.task_timeout_seconds == 11.0
    assert "--artifacts-dir" in options.run_task_command_base
    assert "/tmp/isolation-artifacts" in options.run_task_command_base
    assert "--models" in options.run_task_command_base
    assert "--preprocessing" in options.run_task_command_base
    assert "--no-combined-preprocessing" in options.run_task_command_base
    assert "--no-run-skip" in options.run_task_command_base
    assert "--task-cooldown-seconds" in options.run_task_command_base
    assert "--max-queue-tasks" in options.run_task_command_base
    assert "123456" in options.run_task_command_base
    assert options.max_queue_tasks == 123456
    assert options.queue_export_path == "/tmp/queue-export.jsonl"
    capsys.readouterr()


def test_run_task_command_uses_runner_cooldown_from_config(
    capsys,
    monkeypatch,
) -> None:
    module = _import_run_experiments_module()
    fake_runner = SimpleNamespace(
        config_path=Path("configs/experiment.default.json"),
        project_paths=SimpleNamespace(),
        training_config=SimpleNamespace(),
    )
    captured = {"cooldown": None}

    monkeypatch.setattr(module, "_build_runner", lambda args: fake_runner)
    monkeypatch.setattr(
        module,
        "load_experiment_config",
        lambda _path: SimpleNamespace(
            source_path=Path("configs/experiment.default.json"),
            runner=SimpleNamespace(
                isolate_tasks=False,
                task_cooldown_seconds=3.5,
                task_timeout_seconds=None,
            ),
        ),
    )

    def _fake_run_one_experiment_task(**kwargs):
        captured["cooldown"] = kwargs["task_cooldown_seconds"]
        return {"task_id": kwargs["task_id"], "status": "completed"}

    monkeypatch.setattr(module, "run_one_experiment_task", _fake_run_one_experiment_task)

    exit_code = module.main(["run-task", "--task-id", "exp-cooldown"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Task exp-cooldown completed successfully." in output
    assert captured["cooldown"] == 3.5


def test_run_command_allow_huge_queue_disables_limit(
    capsys,
    monkeypatch,
) -> None:
    module = _import_run_experiments_module()
    captured = {"options": None}

    def _fake_run(options):
        captured["options"] = options
        return {
            "overall_status": "idle",
            "total": 0,
            "counts": {
                "pending": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "stopped": 0,
            },
            "current_task": None,
            "state_path": "/tmp/state.json",
            "summary_path": "/tmp/summary.csv",
            "history_dir": "/tmp/history",
            "predictions_dir": "/tmp/predictions",
            "log_path": "/tmp/runner.log",
            "active_pid": None,
        }

    fake_runner = SimpleNamespace(
        run=_fake_run,
        store=SimpleNamespace(summarize=lambda: {}),
    )

    monkeypatch.setattr(module, "_build_runner", lambda args: fake_runner)
    monkeypatch.setattr(
        module,
        "load_experiment_config",
        lambda _path: SimpleNamespace(
            source_path=Path("configs/experiment.default.json"),
            runner=SimpleNamespace(
                isolate_tasks=False,
                task_cooldown_seconds=2.0,
                task_timeout_seconds=None,
                device_policy="adaptive",
                gpu_retries=1,
                cpu_retries=1,
                cooldown_after_oom_seconds=15.0,
                gpu_recovery_cooldown_seconds=60.0,
                max_consecutive_oom=3,
                max_task_attempts=4,
                fail_fast_on_oom=False,
            ),
        ),
    )

    exit_code = module.main(["run", "--allow-huge-queue"])

    assert exit_code == 0
    options = captured["options"]
    assert options is not None
    assert options.max_queue_tasks is None
    capsys.readouterr()
