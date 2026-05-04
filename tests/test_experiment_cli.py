from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.experiments import ExperimentStateStore
from pipeline.utils.paths import build_project_paths
from pipeline.utils.runtime_limits import CpuExecutionLimits

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
            build_cpu_execution_limits=lambda **kwargs: CpuExecutionLimits(),
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
            "--cpu-max-threads",
            "2",
            "--cpu-opencv-threads",
            "1",
            "--cpu-inter-op-threads",
            "1",
            "--cpu-intra-op-threads",
            "2",
            "--cpu-nice",
            "10",
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
    assert "--cpu-max-threads" in options.run_task_command_base
    assert "--cpu-opencv-threads" in options.run_task_command_base
    assert "--cpu-inter-op-threads" in options.run_task_command_base
    assert "--cpu-intra-op-threads" in options.run_task_command_base
    assert "--cpu-nice" in options.run_task_command_base
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
            build_cpu_execution_limits=lambda **kwargs: CpuExecutionLimits(),
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


def test_run_task_command_forwards_cpu_limits(
    capsys,
    monkeypatch,
) -> None:
    module = _import_run_experiments_module()
    fake_runner = SimpleNamespace(
        config_path=Path("configs/experiment.default.json"),
        project_paths=SimpleNamespace(),
        training_config=SimpleNamespace(),
    )
    captured = {"limits": None}

    monkeypatch.setattr(module, "_build_runner", lambda args: fake_runner)
    monkeypatch.setattr(
        module,
        "load_experiment_config",
        lambda _path: SimpleNamespace(
            source_path=Path("configs/experiment.default.json"),
            build_cpu_execution_limits=lambda **kwargs: CpuExecutionLimits(
                max_threads=kwargs.get("cpu_max_threads"),
                opencv_threads=kwargs.get("cpu_opencv_threads"),
                inter_op_threads=kwargs.get("cpu_inter_op_threads"),
                intra_op_threads=kwargs.get("cpu_intra_op_threads"),
                nice=kwargs.get("cpu_nice"),
            ),
            runner=SimpleNamespace(
                isolate_tasks=False,
                task_cooldown_seconds=3.5,
                task_timeout_seconds=None,
            ),
        ),
    )

    def _fake_run_one_experiment_task(**kwargs):
        captured["limits"] = kwargs["cpu_execution_limits"]
        return {"task_id": kwargs["task_id"], "status": "completed"}

    monkeypatch.setattr(module, "run_one_experiment_task", _fake_run_one_experiment_task)

    exit_code = module.main(
        [
            "run-task",
            "--task-id",
            "exp-cpu-limits",
            "--cpu-max-threads",
            "2",
            "--cpu-opencv-threads",
            "1",
            "--cpu-inter-op-threads",
            "1",
            "--cpu-intra-op-threads",
            "2",
            "--cpu-nice",
            "10",
        ]
    )

    assert exit_code == 0
    assert captured["limits"] == CpuExecutionLimits(
        max_threads=2,
        opencv_threads=1,
        inter_op_threads=1,
        intra_op_threads=2,
        nice=10,
    )
    capsys.readouterr()


def test_probe_runtime_command_emits_json(
    capsys,
    monkeypatch,
) -> None:
    module = _import_run_experiments_module()
    monkeypatch.setattr(
        module,
        "collect_runtime_probe",
        lambda **kwargs: SimpleNamespace(
            ok=True,
            requested_device="cpu",
            effective_device="cpu",
            python_executable=sys.executable,
            tensorflow_imported=True,
            tensorflow_version="test-tf",
            cuda_visible_devices="-1",
            physical_gpu_devices=(),
            logical_gpu_devices=(),
            tensorflow_visible_devices=(),
            nvidia_smi_available=False,
            nvidia_smi_command=(),
            gpu_memory_summary={"devices": []},
            effective_cpu_thread_env={"OMP_NUM_THREADS": "2"},
            tensorflow_tiny_gpu_op=False,
            tensorflow_tiny_gpu_op_device=None,
            gpu_used=False,
            opencv_threads=1,
            warnings=(),
            errors=(),
        ),
    )
    monkeypatch.setattr(
        module,
        "load_experiment_config",
        lambda _path: SimpleNamespace(
            build_cpu_execution_limits=lambda **kwargs: CpuExecutionLimits(max_threads=kwargs.get("cpu_max_threads")),
        ),
    )

    exit_code = module.main(["probe-runtime", "--device", "cpu", "--cpu-max-threads", "2", "--json"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"requested_device": "cpu"' in output
    assert '"effective_cpu_thread_env"' in output


def test_quickstart_contains_common_operations_commands() -> None:
    quickstart = Path("docs/quickstart.md").read_text(encoding="utf-8")

    assert "probe-runtime --device gpu" in quickstart
    assert "probe-runtime --device cpu" in quickstart
    assert "run_experiments.py launch" in quickstart
    assert "run_experiments.py status" in quickstart
    assert "run_experiments.py stop" in quickstart
    assert "run_experiments.py reset --purge-results" in quickstart
    assert "run_experiments.py stop --config configs/experiment.low-memory.json --kill" in quickstart
    assert "run_experiments.py reset --config configs/experiment.low-memory.json --purge-results --kill-active" in quickstart
    assert "tail -f artifacts/experiments/logs/iterative-runner.log" in quickstart


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
            build_cpu_execution_limits=lambda **kwargs: CpuExecutionLimits(),
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


def test_count_command_reports_expected_grid_sizes(capsys) -> None:
    module = _import_run_experiments_module()

    default_payload = module._count_payload(module.build_parser().parse_args(["count"]))
    aug_payload = module._count_payload(
        module.build_parser().parse_args(
            ["count", "--augmentations-per-image", "1", "2", "3"]
        )
    )
    comb_payload = module._count_payload(
        module.build_parser().parse_args(
            ["count", "--combined-preprocessing"]
        )
    )
    both_payload = module._count_payload(
        module.build_parser().parse_args(
            ["count", "--combined-preprocessing", "--augmentations-per-image", "1", "2", "3"]
        )
    )

    assert default_payload["total_experiments"] == 4_330
    assert default_payload["total_fits"] == 17_320
    assert aug_payload["total_experiments"] == 12_990
    assert aug_payload["total_fits"] == 51_960
    assert comb_payload["total_experiments"] == 1_559_530
    assert comb_payload["total_fits"] == 6_238_120
    assert both_payload["total_experiments"] == 4_678_590
    assert both_payload["total_fits"] == 18_714_360

    exit_code = module.main(["count"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Total experiments: 4,330" in output
    assert "Total fits: 17,320" in output


def test_status_auto_discovers_single_active_runner_without_config(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _import_run_experiments_module()
    default_paths = build_project_paths(tmp_path / "raw-data", tmp_path / "artifacts").ensure_artifact_dirs()
    active_paths = build_project_paths(tmp_path / "raw-data", tmp_path / "artifacts-low-memory").ensure_artifact_dirs()
    active_store = ExperimentStateStore(active_paths)
    stdout_path = active_paths.experiment_logs_dir / "background.out.log"
    stderr_path = active_paths.experiment_logs_dir / "background.err.log"
    active_store.write_pid_record(
        config_path=Path("configs/experiment.low-memory.json"),
        command=["python", "run_experiments.py", "run"],
        pid=2222,
        stdout_log_path=stdout_path,
        stderr_log_path=stderr_path,
    )

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "_load_project_paths", lambda args: default_paths)
    monkeypatch.setattr("pipeline.experiments.runner._is_process_alive", lambda pid: int(pid or 0) == 2222)

    exit_code = module.main(["status"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Using the only active runner found under" in output
    assert "Active PID: 2222" in output
    assert "artifacts-low-memory" in output


def test_stop_command_can_kill_active_runner(
    capsys,
    monkeypatch,
) -> None:
    module = _import_run_experiments_module()
    captured = {"force": None}

    def _terminate_active_run(*, force: bool):
        captured["force"] = force
        return 4321

    fake_store = SimpleNamespace(
        has_active_run=lambda: True,
        terminate_active_run=_terminate_active_run,
    )
    monkeypatch.setattr(module, "_resolve_store_for_control_command", lambda args: (fake_store, None))

    exit_code = module.main(["stop", "--kill"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured["force"] is True
    assert "Runner pid=4321 was terminated." in output


def test_reset_command_forwards_kill_active(
    capsys,
    monkeypatch,
) -> None:
    module = _import_run_experiments_module()
    captured = {"purge_results": None, "kill_active": None}

    def _reset(*, purge_results: bool, kill_active: bool):
        captured["purge_results"] = purge_results
        captured["kill_active"] = kill_active

    fake_store = SimpleNamespace(reset=_reset)
    monkeypatch.setattr(module, "_resolve_store_for_control_command", lambda args: (fake_store, None))

    exit_code = module.main(["reset", "--purge-results", "--kill-active"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured == {"purge_results": True, "kill_active": True}
    assert "Runner state and training results were removed." in output
