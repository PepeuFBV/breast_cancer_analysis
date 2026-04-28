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
