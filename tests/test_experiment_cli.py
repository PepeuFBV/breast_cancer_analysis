from __future__ import annotations

import importlib
import os
import sys
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
    }

    module._print_status_snapshot(snapshot)
    output = capsys.readouterr().out
    assert "Background stdout log: /tmp/background.out.log" in output
    assert "Background stderr log: /tmp/background.err.log" in output


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
