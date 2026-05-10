from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pipeline.experiments.failures import classify_task_failure
from pipeline.train.models import MODEL_BUILDERS, ModelBuilder
from pipeline.train.preprocessing import build_preprocessing_task, count_preprocessing_tasks
from pipeline.train.runner import (
    TrainingConfig,
    TrainingTask,
    artifact_paths_for_task,
    build_history_row,
    build_training_tasks,
    iter_training_tasks,
    load_split_dataframe,
    run_training_task,
    save_run_result,
    task_has_existing_artifacts,
)
from pipeline.utils.memory import clear_ml_memory, log_memory_snapshot
from pipeline.utils.paths import ProjectPaths
from pipeline.utils.runtime_limits import CpuExecutionLimits, apply_cpu_runtime_limits, current_cpu_thread_env, resolve_cpu_thread_env
from pipeline.utils.thermal import collect_thermal_snapshot

STATE_SCHEMA_VERSION = 2
RUNNER_STATE_FILENAME = "runner_state.json"
RUNNER_SUMMARY_FILENAME = "experiment_runs.csv"
RUNNER_LOG_FILENAME = "iterative-runner.log"
RUN_EVENTS_FILENAME = "run-events.jsonl"
TASK_LOGS_DIRNAME = "tasks"
RUNNER_PID_FILENAME = "runner_pid.json"
STOP_REQUEST_FILENAME = "stop_requested.flag"
RUNNER_LIFECYCLE_FILENAME = "runner_lifecycle.json"
BACKGROUND_RUNNER_LOG_PREFIX = "background-runner"
TASK_STATUSES = {"pending", "running", "completed", "failed", "stopped"}
MAX_QUEUE_TASKS = 50_000
MATERIALIZED_QUEUE_HARD_LIMIT = 100_000
HUGE_QUEUE_WARNING_TASKS = 25_000
ISOLATED_TASK_CHILD_MODE_ENV = "BREAST_CANCER_ANALYSIS_ISOLATED_TASK_CHILD"
ISOLATED_TASK_PARENT_PID_ENV = "BREAST_CANCER_ANALYSIS_ISOLATED_TASK_PARENT_PID"
REQUESTED_DEVICE_ENV = "BREAST_CANCER_ANALYSIS_REQUESTED_DEVICE"
BACKGROUND_STDOUT_LOG_ENV = "BREAST_CANCER_ANALYSIS_BACKGROUND_STDOUT_LOG_PATH"
BACKGROUND_STDERR_LOG_ENV = "BREAST_CANCER_ANALYSIS_BACKGROUND_STDERR_LOG_PATH"
TEST_TASK_SHIM_PATH_ENV = "BREAST_CANCER_ANALYSIS_TEST_TASK_SHIM_PATH"
DEVICE_POLICY_VALUES = {"gpu-first", "cpu-only", "gpu-only", "adaptive"}
GPU_HEALTH_STATES = {"healthy", "cooling_down", "unhealthy"}
STREAMING_QUEUE_TASK_THRESHOLD = MATERIALIZED_QUEUE_HARD_LIMIT


@dataclass(frozen=True)
class IterativeRunOptions:
    rerun_failed: bool = False
    rerun_completed: bool = False
    limit: int | None = None
    isolate_tasks: bool = False
    task_cooldown_seconds: float = 0.0
    task_timeout_seconds: float | None = None
    run_task_command_base: tuple[str, ...] | None = None
    device_policy: str = "adaptive"
    gpu_retries: int = 1
    cpu_retries: int = 1
    cooldown_after_oom_seconds: float = 15.0
    gpu_recovery_cooldown_seconds: float = 60.0
    max_consecutive_oom: int = 3
    max_task_attempts: int = 4
    fail_fast_on_oom: bool = False
    thermal_policy_enabled: bool = False
    thermal_cpu_temp_celsius_limit: float | None = None
    thermal_cpu_load_percent_limit: float | None = None
    thermal_gpu_temp_celsius_limit: float | None = None
    thermal_gpu_utilization_percent_limit: float | None = None
    thermal_gpu_recovery_temp_celsius: float | None = None
    thermal_cooldown_seconds: float = 30.0
    max_queue_tasks: int | None = MAX_QUEUE_TASKS
    queue_export_path: str | None = None


@dataclass(frozen=True)
class ExperimentGridCounts:
    preprocessing_count: int
    model_count: int
    augmentation_count: int
    total_experiments: int
    total_fits: int
    folds: int


@dataclass(frozen=True)
class BackgroundRunnerLaunch:
    process: subprocess.Popen[Any]
    stdout_path: Path
    stderr_path: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_from_path(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _default_runner_runtime() -> dict[str, Any]:
    return {
        "device_policy": "adaptive",
        "preferred_device": "gpu",
        "gpu_health": "healthy",
        "gpu_oom_count": 0,
        "cpu_fallback_successes": 0,
        "consecutive_final_oom_failures": 0,
        "last_gpu_oom_task_id": None,
        "last_successful_device": None,
        "gpu_recovery_cooldown_until": None,
        "thermal_policy_enabled": False,
        "thermal_state": "disabled",
        "thermal_last_sample_at": None,
        "thermal_last_reason": None,
        "thermal_last_cpu_temp_celsius": None,
        "thermal_last_cpu_load_percent": None,
        "thermal_last_gpu_temp_celsius": None,
        "thermal_last_gpu_utilization_percent": None,
        "oom_policy_stop": None,
        "stream_queue_mode": False,
        "launch_context": {},
    }


def _default_lifecycle_record() -> dict[str, Any]:
    return {
        "desired_state": "running",
        "pause_reason": None,
        "pause_requested_at": None,
        "resume_requested_at": None,
        "last_active_pid": None,
        "last_active_started_at": None,
        "last_active_command": None,
        "last_recovery_reason": None,
        "last_recovery_at": None,
    }


def _apply_task_defaults(task: dict[str, Any]) -> dict[str, Any]:
    task.setdefault("attempt_history", [])
    task.setdefault("gpu_attempts", 0)
    task.setdefault("cpu_attempts", 0)
    task.setdefault("final_device", None)
    task.setdefault("failure_kind", None)
    task.setdefault("fallback_reason", None)
    return task


def _build_background_log_paths(logs_dir: Path) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return (
        logs_dir / f"{BACKGROUND_RUNNER_LOG_PREFIX}-{timestamp}.out.log",
        logs_dir / f"{BACKGROUND_RUNNER_LOG_PREFIX}-{timestamp}.err.log",
    )


def _is_process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        resolved_pid = int(pid)
    except Exception:
        return False
    if resolved_pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            process_handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, resolved_pid)
            if not process_handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if ctypes.windll.kernel32.GetExitCodeProcess(process_handle, ctypes.byref(exit_code)) == 0:
                    return False
                return exit_code.value == 259
            finally:
                ctypes.windll.kernel32.CloseHandle(process_handle)
        except Exception:
            return False
    try:
        os.kill(resolved_pid, 0)
    except OSError:
        return False
    return True


def _terminate_process(pid: int, *, force: bool) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        command = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            command.insert(1, "/F")
        subprocess.run(command, check=False, capture_output=True, text=True)
        return
    try:
        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
    except OSError:
        return


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _normalize_json_value(value.item())
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(_normalize_json_value(payload), indent=2, sort_keys=True),
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_normalize_json_value(payload), sort_keys=True) + "\n")


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_test_task_shim(path: Path) -> dict[str, Any]:
    payload = _load_json_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Test task shim file must contain a JSON object: {path}")
    return payload


def _read_first_csv_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    return {} if row is None else dict(row)


def _queue_signature(tasks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha1(
        json.dumps(
            [
                {
                    "id": task["id"],
                    "model_name": task["model_name"],
                    "preproc_id": task["preproc_id"],
                    "param_id": task["param_id"],
                }
                for task in tasks
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return digest


def _model_runtime_signature(config: TrainingConfig, model_name: str) -> dict[str, Any] | None:
    runtime = (config.model_runtime or {}).get(model_name)
    if runtime is None:
        return None
    return _normalize_json_value(asdict(runtime))


def _stable_path_for_experiment_id(path: Path) -> str:
    """Return an absolute path string without forcing symlink resolution."""

    return str(path.expanduser().absolute())


def build_experiment_id(task: TrainingTask, config: TrainingConfig) -> str:
    payload = {
        "model_name": task.model_name,
        "preproc_id": task.preproc_id,
        "param_id": task.param_id,
        "param_json": json.loads(task.param_json),
        "augmentations_per_image": task.augmentations_per_image,
        "train_split_path": _stable_path_for_experiment_id(config.train_split_path),
        "test_split_path": _stable_path_for_experiment_id(config.test_split_path),
        "folds": config.folds,
        "validation_size": config.validation_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "loss": config.loss,
        "random_state": config.random_state,
        "model_runtime": _model_runtime_signature(config, task.model_name),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"exp-{digest[:16]}"


def build_experiment_record(task: TrainingTask, config: TrainingConfig) -> dict[str, Any]:
    history_path, predictions_path = artifact_paths_for_task(config, task)
    return {
        "id": build_experiment_id(task, config),
        "status": "pending",
        "model_name": task.model_name,
        "preproc_id": task.preproc_id,
        "param_id": task.param_id,
        "param_display": task.param_display,
        "param_json": task.param_json,
        "augmentations_per_image": task.augmentations_per_image,
        "parameters": {
            "model_name": task.model_name,
            "preprocessing_id": task.preproc_id,
            "preprocessing_params": json.loads(task.param_json),
            "augmentations_per_image": task.augmentations_per_image,
            "is_combined": task.is_combined,
            "training": {
                "folds": config.folds,
                "validation_size": config.validation_size,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "loss": config.loss,
                "random_state": config.random_state,
            },
            "model_runtime": _model_runtime_signature(config, task.model_name),
        },
        "artifacts": {
            "history_path": str(history_path),
            "predictions_path": str(predictions_path),
        },
        "created_at": None,
        "updated_at": None,
        "started_at": None,
        "finished_at": None,
        "last_duration_seconds": None,
        "attempts": 0,
        "attempt_history": [],
        "gpu_attempts": 0,
        "cpu_attempts": 0,
        "final_device": None,
        "failure_kind": None,
        "fallback_reason": None,
        "error_summary": None,
        "result_summary": {},
    }


def build_training_task_from_record(
    record: dict[str, Any],
    config: TrainingConfig,
) -> TrainingTask:
    preprocessing_task = build_preprocessing_task(
        str(record["preproc_id"]),
        json.loads(str(record["param_json"])),
        param_grids=config.preprocessing_grids,
    )
    return TrainingTask(
        model_name=str(record["model_name"]),
        preprocessing_task=preprocessing_task,
        augmentations_per_image=int(record.get("augmentations_per_image", 1)),
    )


class ExperimentStateStore:
    def __init__(self, project_paths: ProjectPaths) -> None:
        self.project_paths = project_paths
        self.state_path = project_paths.experiment_state_dir / RUNNER_STATE_FILENAME
        self.summary_path = project_paths.experiment_summary_dir / RUNNER_SUMMARY_FILENAME
        self.log_path = project_paths.experiment_logs_dir / RUNNER_LOG_FILENAME
        self.run_events_path = project_paths.experiment_logs_dir / RUN_EVENTS_FILENAME
        self.task_logs_dir = project_paths.experiment_logs_dir / TASK_LOGS_DIRNAME
        self.pid_path = project_paths.experiment_control_dir / RUNNER_PID_FILENAME
        self.stop_flag_path = project_paths.experiment_control_dir / STOP_REQUEST_FILENAME
        self.lifecycle_path = project_paths.experiment_control_dir / RUNNER_LIFECYCLE_FILENAME

    def ensure_dirs(self) -> None:
        self.project_paths.ensure_artifact_dirs()

    def load_state(self, *, config_path: Path | None = None) -> dict[str, Any]:
        state = _load_json_file(self.state_path) or {}
        state.setdefault("schema_version", STATE_SCHEMA_VERSION)
        state.setdefault("created_at", _timestamp_now())
        state["updated_at"] = state.get("updated_at", state["created_at"])
        state["config_path"] = str(config_path) if config_path else state.get("config_path")
        state["artifacts_dir"] = str(self.project_paths.artifacts_dir)
        state["history_dir"] = str(self.project_paths.history_dir)
        state["predictions_dir"] = str(self.project_paths.predictions_dir)
        state["summary_path"] = str(self.summary_path)
        state["current_task_id"] = state.get("current_task_id")
        state["tasks"] = [_apply_task_defaults(dict(task)) for task in list(state.get("tasks", []))]
        runtime = dict(_default_runner_runtime())
        runtime.update(dict(state.get("runtime", {})))
        state["runtime"] = runtime
        return state

    def read_pid_record(self) -> dict[str, Any] | None:
        return _load_json_file(self.pid_path)

    def read_lifecycle_record(self) -> dict[str, Any]:
        payload = dict(_default_lifecycle_record())
        payload.update(dict(_load_json_file(self.lifecycle_path) or {}))
        desired_state = str(payload.get("desired_state", "running"))
        if desired_state not in {"running", "paused"}:
            desired_state = "running"
        payload["desired_state"] = desired_state
        return payload

    def _write_lifecycle_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(_default_lifecycle_record())
        resolved.update(_normalize_json_value(payload))
        desired_state = str(resolved.get("desired_state", "running"))
        if desired_state not in {"running", "paused"}:
            raise ValueError("desired_state must be either 'running' or 'paused'.")
        resolved["desired_state"] = desired_state
        _atomic_write_json(self.lifecycle_path, resolved)
        return resolved

    def update_lifecycle_record(self, updates: dict[str, Any]) -> dict[str, Any]:
        payload = self.read_lifecycle_record()
        payload.update(_normalize_json_value(updates))
        return self._write_lifecycle_record(payload)

    def set_desired_state(
        self,
        desired_state: str,
        *,
        reason: str | None = None,
        requested_at: str | None = None,
    ) -> dict[str, Any]:
        now = requested_at or _timestamp_now()
        if desired_state == "paused":
            return self.update_lifecycle_record(
                {
                    "desired_state": "paused",
                    "pause_reason": (reason or "manual"),
                    "pause_requested_at": now,
                }
            )
        if desired_state == "running":
            return self.update_lifecycle_record(
                {
                    "desired_state": "running",
                    "resume_requested_at": now,
                    "pause_reason": None,
                    "pause_requested_at": None,
                }
            )
        raise ValueError("desired_state must be either 'running' or 'paused'.")

    def record_recovery_event(self, *, reason: str) -> dict[str, Any]:
        return self.update_lifecycle_record(
            {
                "last_recovery_reason": reason,
                "last_recovery_at": _timestamp_now(),
            }
        )

    def clear_lifecycle_record(self) -> None:
        if self.lifecycle_path.exists():
            self.lifecycle_path.unlink()

    def has_active_run(self) -> bool:
        pid_record = self.read_pid_record()
        pid = None if pid_record is None else pid_record.get("pid")
        return _is_process_alive(pid)

    def terminate_active_run(
        self,
        *,
        force: bool,
        wait_seconds: float = 15.0,
    ) -> int | None:
        pid_record = self.read_pid_record() or {}
        pid = pid_record.get("pid")
        try:
            resolved_pid = None if pid is None else int(pid)
        except Exception:
            resolved_pid = None
        if resolved_pid is None:
            self.clear_pid_record()
            return None
        if not _is_process_alive(resolved_pid):
            self.clear_pid_record()
            return resolved_pid

        _terminate_process(resolved_pid, force=force)
        deadline = time.time() + max(0.0, float(wait_seconds))
        while time.time() < deadline:
            if not _is_process_alive(resolved_pid):
                self.clear_pid_record()
                return resolved_pid
            time.sleep(0.2)
        if not _is_process_alive(resolved_pid):
            self.clear_pid_record()
            return resolved_pid
        raise RuntimeError(f"Could not stop runner pid={resolved_pid}. " "Try closing the process manually and rerun the command.")

    def write_pid_record(
        self,
        *,
        config_path: Path,
        command: list[str],
        pid: int | None = None,
        stdout_log_path: Path | str | None = None,
        stderr_log_path: Path | str | None = None,
    ) -> None:
        resolved_pid = os.getpid() if pid is None else int(pid)
        existing = self.read_pid_record() or {}
        if existing.get("pid") != resolved_pid:
            existing = {}
        resolved_stdout_log = existing.get("stdout_log_path") if stdout_log_path is None else stdout_log_path
        resolved_stderr_log = existing.get("stderr_log_path") if stderr_log_path is None else stderr_log_path

        payload: dict[str, Any] = {
            "pid": resolved_pid,
            "config_path": str(config_path),
            "command": command,
            "started_at": _timestamp_now(),
        }
        if resolved_stdout_log is not None:
            payload["stdout_log_path"] = str(resolved_stdout_log)
        if resolved_stderr_log is not None:
            payload["stderr_log_path"] = str(resolved_stderr_log)

        _atomic_write_json(self.pid_path, payload)
        self.update_lifecycle_record(
            {
                "last_active_pid": resolved_pid,
                "last_active_started_at": payload["started_at"],
                "last_active_command": list(command),
            }
        )

    def clear_pid_record(self) -> None:
        if self.pid_path.exists():
            self.pid_path.unlink()

    def request_stop(self, *, reason: str = "manual") -> Path:
        requested_at = _timestamp_now()
        self.set_desired_state(
            "paused",
            reason=reason,
            requested_at=requested_at,
        )
        _atomic_write_text(
            self.stop_flag_path,
            json.dumps({"requested_at": requested_at, "reason": reason}),
        )
        return self.stop_flag_path

    def clear_stop_request(self) -> None:
        if self.stop_flag_path.exists():
            self.stop_flag_path.unlink()

    def stop_requested(self) -> bool:
        return self.stop_flag_path.exists()

    def mark_launch_requested(self) -> dict[str, Any]:
        self.clear_stop_request()
        return self.set_desired_state("running", requested_at=_timestamp_now())

    def reconcile_for_launch(self) -> dict[str, Any] | None:
        pid_record = self.read_pid_record()
        if pid_record is None:
            return None
        stale_pid = pid_record.get("pid")
        if _is_process_alive(stale_pid):
            return None

        stale_state = self.load_state()
        running_before = sum(1 for task in stale_state["tasks"] if task.get("status") == "running")
        if running_before > 0:
            self._reconcile_running_tasks(stale_state)
            self._persist_state(stale_state, full_snapshot_sync=True)
        self.clear_pid_record()
        reason = f"Recovered from stale runner pid={stale_pid}; marked {running_before} running task(s) as stopped." if running_before > 0 else f"Recovered from stale runner pid={stale_pid}; no running tasks required reconciliation."
        self.record_recovery_event(reason=reason)
        return {
            "stale_pid": stale_pid,
            "recovered_running_tasks": running_before,
            "reason": reason,
        }

    def sync_queue(
        self,
        queue_records: list[dict[str, Any]],
        *,
        config_path: Path,
    ) -> dict[str, Any]:
        state = self.load_state(config_path=config_path)
        existing_by_id = {task["id"]: task for task in state["tasks"] if isinstance(task, dict) and task.get("id")}
        now = _timestamp_now()

        synced_tasks: list[dict[str, Any]] = []
        for record in queue_records:
            existing = existing_by_id.get(record["id"])
            if existing is None:
                synced_tasks.append(
                    {
                        **record,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                continue

            synced_tasks.append(
                {
                    **record,
                    "status": existing.get("status", "pending"),
                    "created_at": existing.get("created_at", now),
                    "updated_at": existing.get("updated_at", now),
                    "started_at": existing.get("started_at"),
                    "finished_at": existing.get("finished_at"),
                    "last_duration_seconds": existing.get("last_duration_seconds"),
                    "attempts": int(existing.get("attempts", 0)),
                    "attempt_history": list(existing.get("attempt_history", [])),
                    "gpu_attempts": int(existing.get("gpu_attempts", 0)),
                    "cpu_attempts": int(existing.get("cpu_attempts", 0)),
                    "final_device": existing.get("final_device"),
                    "failure_kind": existing.get("failure_kind"),
                    "fallback_reason": existing.get("fallback_reason"),
                    "error_summary": existing.get("error_summary"),
                    "result_summary": dict(existing.get("result_summary", {})),
                }
            )

        state["tasks"] = synced_tasks
        state["config_path"] = str(config_path)
        state["queue_signature"] = _queue_signature(synced_tasks)
        state["updated_at"] = now
        state["current_task_id"] = None
        runtime = dict(_default_runner_runtime())
        runtime.update(dict(state.get("runtime", {})))
        state["runtime"] = runtime

        self._reconcile_running_tasks(state)
        self._reconcile_artifacts(state)
        self._persist_state(state, full_snapshot_sync=True)
        return state

    def register_task_record(
        self,
        record: dict[str, Any],
        *,
        config_path: Path,
    ) -> dict[str, Any]:
        state = self.load_state(config_path=config_path)
        existing = next((task for task in state["tasks"] if task.get("id") == record["id"]), None)
        now = _timestamp_now()
        if existing is None:
            task_payload = {
                **record,
                "created_at": now,
                "updated_at": now,
            }
            state["tasks"].append(_apply_task_defaults(task_payload))
        else:
            existing.update(
                {
                    **record,
                    "created_at": existing.get("created_at", now),
                    "updated_at": now,
                    "started_at": existing.get("started_at"),
                    "finished_at": existing.get("finished_at"),
                    "last_duration_seconds": existing.get("last_duration_seconds"),
                    "attempts": int(existing.get("attempts", 0)),
                    "attempt_history": list(existing.get("attempt_history", [])),
                    "gpu_attempts": int(existing.get("gpu_attempts", 0)),
                    "cpu_attempts": int(existing.get("cpu_attempts", 0)),
                    "final_device": existing.get("final_device"),
                    "failure_kind": existing.get("failure_kind"),
                    "fallback_reason": existing.get("fallback_reason"),
                    "error_summary": existing.get("error_summary"),
                    "result_summary": dict(existing.get("result_summary", {})),
                }
            )
            _apply_task_defaults(existing)
        self._persist_state(state, snapshot_task_ids={str(record["id"])})
        return state

    def _reconcile_running_tasks(self, state: dict[str, Any]) -> None:
        now = _timestamp_now()
        for task in state["tasks"]:
            if task.get("status") == "running":
                task["status"] = "stopped"
                task["updated_at"] = now
                task["finished_at"] = task.get("finished_at") or now
                task["error_summary"] = task.get("error_summary") or "Runner interrupted before experiment completion."

    def _reconcile_artifacts(self, state: dict[str, Any]) -> None:
        now = _timestamp_now()
        for task in state["tasks"]:
            artifacts = task.get("artifacts", {})
            history_path = Path(artifacts.get("history_path", ""))
            predictions_path = Path(artifacts.get("predictions_path", ""))
            artifacts_exist = history_path.exists() and predictions_path.exists()

            if task.get("status") == "completed" and not artifacts_exist:
                task["status"] = "pending"
                task["updated_at"] = now
                task["finished_at"] = None
                task["error_summary"] = "Persisted state referenced missing artifacts; task re-queued."
                task["result_summary"] = {}
                continue

            if not artifacts_exist:
                continue

            if task.get("status") in {"pending", "running", "stopped"}:
                task["status"] = "completed"
                task["updated_at"] = now
                task["finished_at"] = task.get("finished_at") or _timestamp_from_path(history_path)
                task["attempts"] = max(1, int(task.get("attempts", 0)))
                task["error_summary"] = None
                task["result_summary"] = _read_first_csv_row(history_path)

    def _persist_state(
        self,
        state: dict[str, Any],
        *,
        full_snapshot_sync: bool = False,
        snapshot_task_ids: set[str] | None = None,
    ) -> None:
        state["updated_at"] = _timestamp_now()
        _atomic_write_json(self.state_path, state)
        stream_queue_mode = bool(dict(state.get("runtime", {})).get("stream_queue_mode"))
        if not stream_queue_mode:
            self._write_summary_csv(state)
        if full_snapshot_sync:
            self._write_task_snapshots(state)
        elif snapshot_task_ids:
            self._write_task_snapshots_for_ids(state, snapshot_task_ids)

    def _write_task_snapshots(self, state: dict[str, Any]) -> None:
        self.project_paths.experiment_task_dir.mkdir(parents=True, exist_ok=True)
        active_task_ids = set()
        for task in state["tasks"]:
            task_id = task["id"]
            active_task_ids.add(task_id)
            _atomic_write_json(
                self.project_paths.experiment_task_dir / f"{task_id}.json",
                task,
            )

        for snapshot_path in self.project_paths.experiment_task_dir.glob("*.json"):
            if snapshot_path.stem not in active_task_ids:
                snapshot_path.unlink()

    def _write_task_snapshots_for_ids(
        self,
        state: dict[str, Any],
        task_ids: set[str],
    ) -> None:
        if not task_ids:
            return
        self.project_paths.experiment_task_dir.mkdir(parents=True, exist_ok=True)
        for task in state["tasks"]:
            if task["id"] not in task_ids:
                continue
            _atomic_write_json(
                self.project_paths.experiment_task_dir / f"{task['id']}.json",
                task,
            )

    def _write_summary_csv(self, state: dict[str, Any]) -> None:
        rows = [self._summary_row_from_task(task) for task in state["tasks"]]
        fieldnames = sorted({key for row in rows for key in row})
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            "w",
            dir=self.summary_path.parent,
            delete=False,
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            temp_path = Path(handle.name)
        temp_path.replace(self.summary_path)

    def append_summary_row(self, task: dict[str, Any]) -> None:
        row = self._summary_row_from_task(task)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.summary_path.exists():
            with self.summary_path.open("w", encoding="utf-8", newline="") as handle:
                fieldnames = sorted(row.keys())
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        with self.summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = None
            missing_keys = [key for key in row.keys() if key not in existing_fieldnames]
            if missing_keys:
                existing_rows = list(reader)

        if missing_keys:
            fieldnames = sorted({*existing_fieldnames, *row.keys()})
            with tempfile.NamedTemporaryFile(
                "w",
                dir=self.summary_path.parent,
                delete=False,
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for existing_row in existing_rows or []:
                    writer.writerow(existing_row)
                writer.writerow(row)
                temp_path = Path(handle.name)
            temp_path.replace(self.summary_path)
            return

        with self.summary_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=existing_fieldnames)
            writer.writerow(row)

    def _summary_row_from_task(self, task: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": task["id"],
            "status": task["status"],
            "model_name": task["model_name"],
            "preproc_id": task["preproc_id"],
            "param_id": task["param_id"],
            "param_display": task["param_display"],
            "param_json": task["param_json"],
            "attempts": task.get("attempts", 0),
            "gpu_attempts": task.get("gpu_attempts", 0),
            "cpu_attempts": task.get("cpu_attempts", 0),
            "final_device": task.get("final_device"),
            "failure_kind": task.get("failure_kind"),
            "fallback_reason": task.get("fallback_reason"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
            "updated_at": task.get("updated_at"),
            "last_duration_seconds": task.get("last_duration_seconds"),
            "error_summary": task.get("error_summary"),
            "history_path": task.get("artifacts", {}).get("history_path"),
            "predictions_path": task.get("artifacts", {}).get("predictions_path"),
            "parameters_json": json.dumps(task.get("parameters", {}), sort_keys=True, separators=(",", ":")),
        }
        row.update({f"result_{key}": value for key, value in task.get("result_summary", {}).items()})
        return row

    def select_runnable_task_ids(
        self,
        state: dict[str, Any],
        *,
        rerun_failed: bool,
        rerun_completed: bool,
    ) -> list[str]:
        selected: list[str] = []
        for task in state["tasks"]:
            status = task.get("status", "pending")
            if status in {"pending", "stopped"}:
                selected.append(task["id"])
                continue
            if status == "failed" and rerun_failed:
                selected.append(task["id"])
                continue
            if status == "completed" and rerun_completed:
                selected.append(task["id"])
        return selected

    def update_task_status(
        self,
        task_id: str,
        *,
        status: str,
        error_summary: str | None = None,
        result_summary: dict[str, Any] | None = None,
        duration_seconds: float | None = None,
    ) -> dict[str, Any]:
        if status not in TASK_STATUSES:
            raise ValueError(f"Unsupported task status: {status}")

        state = self.load_state()
        stream_queue_mode = bool(dict(state.get("runtime", {})).get("stream_queue_mode"))
        expected_counts = state.get("expected_counts")
        now = _timestamp_now()
        task_found = False
        for task in state["tasks"]:
            if task["id"] != task_id:
                continue

            task_found = True
            previous_status = str(task.get("status", "pending"))
            task["status"] = status
            task["updated_at"] = now
            task["error_summary"] = error_summary
            task["last_duration_seconds"] = duration_seconds

            if status == "running":
                task["started_at"] = now
                task["finished_at"] = None
                task["attempts"] = int(task.get("attempts", 0)) + 1
                task["result_summary"] = {}
                state["current_task_id"] = task_id
            else:
                task["finished_at"] = now
                if result_summary is not None:
                    task["result_summary"] = _normalize_json_value(result_summary)
                state["current_task_id"] = None

            if stream_queue_mode and isinstance(expected_counts, dict) and previous_status != status:
                if previous_status in TASK_STATUSES:
                    expected_counts[previous_status] = max(0, int(expected_counts.get(previous_status, 0)) - 1)
                expected_counts[status] = int(expected_counts.get(status, 0)) + 1
            break

        if not task_found:
            raise KeyError(f"Task id not found in runner state: {task_id}")

        if stream_queue_mode and status in {"completed", "failed", "stopped"}:
            state["expected_counts"] = expected_counts
            self.append_summary_row(next(task for task in state["tasks"] if task["id"] == task_id))
        self._persist_state(state, snapshot_task_ids={task_id})
        return state

    def append_task_attempt(self, task_id: str, attempt_record: dict[str, Any]) -> dict[str, Any]:
        state = self.load_state()
        task_found = False
        for task in state["tasks"]:
            if task["id"] != task_id:
                continue
            task_found = True
            _apply_task_defaults(task)
            history = list(task.get("attempt_history", []))
            history.append(_normalize_json_value(attempt_record))
            task["attempt_history"] = history
            device = str(attempt_record.get("device", "unknown"))
            if device == "gpu":
                task["gpu_attempts"] = int(task.get("gpu_attempts", 0)) + 1
            elif device == "cpu":
                task["cpu_attempts"] = int(task.get("cpu_attempts", 0)) + 1
            failure_kind = attempt_record.get("failure_kind")
            if failure_kind:
                task["failure_kind"] = failure_kind
            task["updated_at"] = _timestamp_now()
            break
        if not task_found:
            raise KeyError(f"Task id not found in runner state: {task_id}")
        self._persist_state(state, snapshot_task_ids={task_id})
        return state

    def update_task_execution_details(
        self,
        task_id: str,
        *,
        final_device: str | None = None,
        failure_kind: str | None = None,
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        state = self.load_state()
        task_found = False
        for task in state["tasks"]:
            if task["id"] != task_id:
                continue
            task_found = True
            _apply_task_defaults(task)
            if final_device is not None:
                task["final_device"] = final_device
            if failure_kind is not None:
                task["failure_kind"] = failure_kind
            if fallback_reason is not None:
                task["fallback_reason"] = fallback_reason
            task["updated_at"] = _timestamp_now()
            break
        if not task_found:
            raise KeyError(f"Task id not found in runner state: {task_id}")
        self._persist_state(state, snapshot_task_ids={task_id})
        return state

    def update_runtime(self, updates: dict[str, Any]) -> dict[str, Any]:
        state = self.load_state()
        runtime = dict(_default_runner_runtime())
        runtime.update(dict(state.get("runtime", {})))
        runtime.update(_normalize_json_value(updates))
        state["runtime"] = runtime
        self._persist_state(state)
        return state

    def set_task_artifacts(self, task_id: str, *, history_path: Path, predictions_path: Path) -> dict[str, Any]:
        state = self.load_state()
        task_found = False
        for task in state["tasks"]:
            if task["id"] == task_id:
                task_found = True
                task["artifacts"] = {
                    "history_path": str(history_path),
                    "predictions_path": str(predictions_path),
                }
                task["updated_at"] = _timestamp_now()
                break
        if not task_found:
            raise KeyError(f"Task id not found in runner state: {task_id}")
        self._persist_state(state, snapshot_task_ids={task_id})
        return state

    def set_expected_queue_totals(self, *, total_experiments: int) -> dict[str, Any]:
        state = self.load_state()
        resolved_total = max(0, int(total_experiments))
        state["expected_total_experiments"] = resolved_total
        state["expected_counts"] = {
            "pending": resolved_total,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "stopped": 0,
        }
        self._persist_state(state)
        return state

    def summarize(self) -> dict[str, Any]:
        state = self.load_state()
        lifecycle = self.read_lifecycle_record()
        pid_record = self.read_pid_record()
        active_pid = None if pid_record is None else pid_record.get("pid")
        stdout_log_path = None if pid_record is None else pid_record.get("stdout_log_path")
        stderr_log_path = None if pid_record is None else pid_record.get("stderr_log_path")
        active_run = _is_process_alive(active_pid)
        if not active_run and any(task.get("status") == "running" for task in state["tasks"]):
            self._reconcile_running_tasks(state)
            self._persist_state(state, full_snapshot_sync=True)

        counts = {status: 0 for status in TASK_STATUSES}
        for task in state["tasks"]:
            status = task.get("status", "pending")
            counts[status] = counts.get(status, 0) + 1

        current_task = None
        for task in state["tasks"]:
            if task["id"] == state.get("current_task_id"):
                current_task = task
                break

        runtime = dict(_default_runner_runtime())
        runtime.update(dict(state.get("runtime", {})))
        total = len(state["tasks"])
        expected_total = state.get("expected_total_experiments")
        expected_counts = state.get("expected_counts")
        stream_queue_mode = bool(runtime.get("stream_queue_mode"))
        if (total == 0 or stream_queue_mode) and isinstance(expected_total, int) and expected_total > 0 and isinstance(expected_counts, dict):
            total = int(expected_total)
            counts = {status: int(expected_counts.get(status, 0)) for status in TASK_STATUSES}
        stop_requested = self.stop_requested()
        desired_state = str(lifecycle.get("desired_state", "running"))
        paused_with_remaining_work = desired_state == "paused" and (counts["pending"] > 0 or counts["stopped"] > 0 or stop_requested)
        if active_run and stop_requested:
            overall_status = "stopping"
        elif active_run:
            overall_status = "running"
        elif paused_with_remaining_work:
            overall_status = "paused"
        elif counts["running"] > 0:
            overall_status = "stopped"
        elif stop_requested and (counts["pending"] > 0 or counts["stopped"] > 0):
            overall_status = "stopped"
        elif counts["pending"] > 0 or counts["stopped"] > 0:
            overall_status = "idle"
        elif counts["failed"] > 0 and counts["completed"] > 0:
            overall_status = "completed_with_failures"
        elif counts["failed"] == total and total > 0:
            overall_status = "failed"
        elif counts["completed"] == total and total > 0:
            overall_status = "completed"
        else:
            overall_status = "idle"
        return {
            "overall_status": overall_status,
            "counts": counts,
            "total": total,
            "current_task": current_task,
            "state_path": str(self.state_path),
            "summary_path": str(self.summary_path),
            "results_root": str(self.project_paths.artifacts_dir),
            "history_dir": str(self.project_paths.history_dir),
            "predictions_dir": str(self.project_paths.predictions_dir),
            "log_path": str(self.log_path),
            "stop_requested": stop_requested,
            "active_pid": active_pid if active_run else None,
            "background_stdout_log_path": stdout_log_path,
            "background_stderr_log_path": stderr_log_path,
            "config_path": state.get("config_path"),
            "updated_at": state.get("updated_at"),
            "desired_state": desired_state,
            "pause_reason": lifecycle.get("pause_reason"),
            "pause_requested_at": lifecycle.get("pause_requested_at"),
            "resume_requested_at": lifecycle.get("resume_requested_at"),
            "last_active_pid": lifecycle.get("last_active_pid"),
            "last_active_started_at": lifecycle.get("last_active_started_at"),
            "last_active_command": lifecycle.get("last_active_command"),
            "last_recovery_reason": lifecycle.get("last_recovery_reason"),
            "last_recovery_at": lifecycle.get("last_recovery_at"),
            "device_policy": runtime.get("device_policy"),
            "preferred_device": runtime.get("preferred_device"),
            "gpu_health": runtime.get("gpu_health"),
            "last_gpu_oom_task_id": runtime.get("last_gpu_oom_task_id"),
            "gpu_oom_count": runtime.get("gpu_oom_count"),
            "cpu_fallback_successes": runtime.get("cpu_fallback_successes"),
            "consecutive_final_oom_failures": runtime.get("consecutive_final_oom_failures"),
            "last_successful_device": runtime.get("last_successful_device"),
            "oom_policy_stop": runtime.get("oom_policy_stop"),
            "thermal_policy_enabled": runtime.get("thermal_policy_enabled"),
            "thermal_state": runtime.get("thermal_state"),
            "thermal_last_sample_at": runtime.get("thermal_last_sample_at"),
            "thermal_last_reason": runtime.get("thermal_last_reason"),
            "thermal_last_cpu_temp_celsius": runtime.get("thermal_last_cpu_temp_celsius"),
            "thermal_last_cpu_load_percent": runtime.get("thermal_last_cpu_load_percent"),
            "thermal_last_gpu_temp_celsius": runtime.get("thermal_last_gpu_temp_celsius"),
            "thermal_last_gpu_utilization_percent": runtime.get("thermal_last_gpu_utilization_percent"),
            "isolate_tasks": runtime.get("isolate_tasks"),
            "allow_huge_queue": runtime.get("allow_huge_queue"),
            "max_queue_tasks": runtime.get("max_queue_tasks"),
            "stream_queue_mode": runtime.get("stream_queue_mode"),
        }

    def partial_results(
        self,
        *,
        max_rows: int = 500,
        all_rows: bool = False,
    ) -> dict[str, Any]:
        if max_rows <= 0:
            raise ValueError("max_rows must be > 0.")

        snapshot = self.summarize()
        state = self.load_state()
        runtime = dict(_default_runner_runtime())
        runtime.update(dict(state.get("runtime", {})))
        launch_context = dict(runtime.get("launch_context", {}))

        skip_reason_counts = dict(launch_context.get("skip_reason_counts", {}))
        skipped_count = int(launch_context.get("skipped_count", 0))
        counts = dict(snapshot.get("counts", {}))
        counts["skipped"] = skipped_count

        rows, matched_count = self._read_partial_summary_rows(
            max_rows=max_rows,
            all_rows=all_rows,
        )
        capped = (not all_rows) and matched_count > len(rows)
        return {
            "schema_version": int(state.get("schema_version", STATE_SCHEMA_VERSION)),
            "generated_at": _timestamp_now(),
            "overall_status": snapshot.get("overall_status"),
            "total": int(snapshot.get("total", 0)),
            "counts": counts,
            "rows_returned": len(rows),
            "rows_capped": capped,
            "row_filter": "finalized_only",
            "paths": {
                "state_path": str(self.state_path),
                "summary_path": str(self.summary_path),
                "history_dir": str(self.project_paths.history_dir),
                "predictions_dir": str(self.project_paths.predictions_dir),
                "log_path": str(self.log_path),
            },
            "launch": {
                "launch_id": launch_context.get("launch_id"),
                "started_at": launch_context.get("started_at"),
                "stream_queue_mode": bool(launch_context.get("stream_queue_mode", runtime.get("stream_queue_mode", False))),
                "rerun_failed": bool(launch_context.get("rerun_failed", False)),
                "rerun_completed": bool(launch_context.get("rerun_completed", False)),
                "limit": launch_context.get("limit"),
                "planned_runnable_count": launch_context.get("planned_runnable_count"),
                "skipped_count": skipped_count,
                "skip_reason_counts": {
                    "already_completed": int(skip_reason_counts.get("already_completed", 0)),
                    "failed_without_rerun": int(skip_reason_counts.get("failed_without_rerun", 0)),
                    "limit_excluded": int(skip_reason_counts.get("limit_excluded", 0)),
                },
            },
            "partial_rows": rows,
        }

    def _read_partial_summary_rows(
        self,
        *,
        max_rows: int,
        all_rows: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self.summary_path.exists():
            return [], 0
        finalized_statuses = {"completed", "failed", "stopped"}
        if all_rows:
            rows: list[dict[str, Any]] = []
            with self.summary_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("status")) in finalized_statuses:
                        rows.append(dict(row))
            return rows, len(rows)

        latest_rows = deque(maxlen=max_rows)
        matched_count = 0
        with self.summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("status")) in finalized_statuses:
                    matched_count += 1
                    latest_rows.append(dict(row))
        return list(latest_rows), matched_count

    def reset(
        self,
        *,
        purge_results: bool = False,
        kill_active: bool = False,
    ) -> None:
        if self.has_active_run():
            pid_record = self.read_pid_record() or {}
            pid = pid_record.get("pid")
            if not kill_active:
                raise RuntimeError(f"Runner is still active with pid={pid}. " "Use `run_experiments.py stop` for a graceful stop, or " "`run_experiments.py reset --kill-active` to stop it before reset.")
            self.terminate_active_run(force=True)
        self.clear_stop_request()
        self.clear_pid_record()
        self.clear_lifecycle_record()

        for path in (self.state_path, self.summary_path):
            if path.exists():
                path.unlink()

        if self.project_paths.experiment_task_dir.exists():
            shutil.rmtree(self.project_paths.experiment_task_dir)

        if purge_results:
            for directory in (
                self.project_paths.history_dir,
                self.project_paths.predictions_dir,
                self.project_paths.experiments_dir,
            ):
                if directory.exists():
                    shutil.rmtree(directory)

        self.ensure_dirs()


class _AppendFileHandler(logging.Handler):
    terminator = "\n"

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(message + self.terminator)
        except Exception:
            self.handleError(record)


class IterativeExperimentRunner:
    def __init__(
        self,
        *,
        config_path: Path,
        project_paths: ProjectPaths,
        training_config: TrainingConfig,
        model_builders: dict[str, ModelBuilder] | None = None,
        preprocessing_tasks: Iterable[Any] | None = None,
        cpu_execution_limits: CpuExecutionLimits | None = None,
    ) -> None:
        self.config_path = config_path
        self.project_paths = project_paths
        self.training_config = training_config
        self.model_builders = model_builders
        self.preprocessing_tasks = preprocessing_tasks
        self.cpu_execution_limits = cpu_execution_limits or CpuExecutionLimits()
        self.store = ExperimentStateStore(project_paths)
        self.logger = self._build_logger()
        self._previous_signal_handlers: dict[int, Any] = {}
        self._peak_process_memory_mb: float | None = None
        self._task_cooldown_seconds: float = 0.0
        self._cpu_limits_applied = False

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"iterative_experiment_runner:{self.project_paths.artifacts_dir}")
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        logger.propagate = False
        self.store.log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = _AppendFileHandler(self.store.log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger

    def _close_logger(self) -> None:
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

    def __del__(self) -> None:
        try:
            self._close_logger()
        except Exception:
            pass

    @staticmethod
    def _pid_matches_current_process(pid: object) -> bool:
        if pid is None:
            return False
        try:
            return int(pid) == os.getpid()
        except Exception:
            return False

    def _capture_memory_snapshot(self, label: str) -> dict[str, Any]:
        snapshot = log_memory_snapshot(label, logger=self.logger)
        process_memory_mb = snapshot.get("process_memory_mb")
        if isinstance(process_memory_mb, (int, float)):
            if self._peak_process_memory_mb is None or process_memory_mb > self._peak_process_memory_mb:
                self._peak_process_memory_mb = float(process_memory_mb)
        return snapshot

    def _task_log_paths(self, task_id: str) -> tuple[Path, Path, Path]:
        return (
            self.store.task_logs_dir / f"{task_id}.events.jsonl",
            self.store.task_logs_dir / f"{task_id}.memory.jsonl",
            self.store.task_logs_dir / f"{task_id}.log",
        )

    def _append_task_log_line(
        self,
        task_id: str,
        *,
        timestamp: str,
        phase: str,
        attempt: int | None,
        process_memory_mb: float | int | None,
        message: str | None,
        error_type: str | None,
        error_message: str | None,
    ) -> None:
        _, _, task_log_path = self._task_log_paths(task_id)
        line = f"{timestamp} phase={phase} attempt={attempt} " f"process_memory_mb={process_memory_mb}"
        if message:
            line = f"{line} message={message}"
        if error_type:
            line = f"{line} error_type={error_type}"
        if error_message:
            line = f"{line} error_message={error_message}"
        with task_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _traceback_summary(self, error: BaseException, *, limit: int = 20) -> str:
        summary = "".join(
            traceback.format_exception(
                type(error),
                error,
                error.__traceback__,
                limit=limit,
            )
        )
        return summary[:8_000]

    def _log_structured_phase(
        self,
        *,
        phase: str,
        task_record: dict[str, Any] | None = None,
        attempt: int | None = None,
        message: str | None = None,
        error: BaseException | None = None,
        traceback_summary: str | None = None,
        event: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = self._capture_memory_snapshot(phase)
        timestamp = _timestamp_now()
        task_id = None if task_record is None else task_record.get("id")
        resolved_error_type = None if error is None else error.__class__.__name__
        resolved_error_message = None if error is None else str(error)

        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "event": event or phase,
            "task_id": task_id,
            "model_name": (None if task_record is None else task_record.get("model_name")),
            "preproc_id": (None if task_record is None else task_record.get("preproc_id")),
            "param_id": None if task_record is None else task_record.get("param_id"),
            "phase": phase,
            "attempt": attempt,
            "process_memory_mb": snapshot.get("process_memory_mb"),
            "peak_process_memory_mb": snapshot.get("peak_process_memory_mb"),
            "gpu_memory": snapshot.get("gpu_memory") or {},
            "tf_memory": snapshot.get("tf_memory") or {},
            "message": message,
            "error_type": resolved_error_type,
            "error_message": resolved_error_message,
        }
        if traceback_summary:
            payload["traceback_summary"] = traceback_summary
        if extra:
            payload.update(_normalize_json_value(extra))

        _append_jsonl(self.store.run_events_path, payload)

        if task_id:
            task_events_path, task_memory_path, _ = self._task_log_paths(task_id)
            _append_jsonl(task_events_path, payload)
            _append_jsonl(
                task_memory_path,
                {
                    "timestamp": timestamp,
                    "task_id": task_id,
                    "phase": phase,
                    "attempt": attempt,
                    "process_memory_mb": snapshot.get("process_memory_mb"),
                    "peak_process_memory_mb": snapshot.get("peak_process_memory_mb"),
                    "gpu_memory": snapshot.get("gpu_memory") or {},
                    "tf_memory": snapshot.get("tf_memory") or {},
                },
            )
            self._append_task_log_line(
                task_id,
                timestamp=timestamp,
                phase=phase,
                attempt=attempt,
                process_memory_mb=snapshot.get("process_memory_mb"),
                message=message,
                error_type=resolved_error_type,
                error_message=resolved_error_message,
            )
        return payload

    def build_queue(self) -> list[tuple[dict[str, Any], TrainingTask]]:
        return self.build_queue_with_options()

    def estimate_grid_counts(
        self,
        *,
        resolved_preprocessing_tasks: Iterable[Any] | None = None,
    ) -> ExperimentGridCounts:
        available_builders = self.model_builders or MODEL_BUILDERS
        model_names = self.training_config.model_names or list(available_builders.keys())
        augmentation_values = tuple(int(value) for value in self.training_config.augmentation_values)
        augmentation_count = len(augmentation_values)
        if augmentation_count <= 0:
            raise ValueError("augmentation_values cannot be empty.")

        preprocessing_count = (
            len(list(resolved_preprocessing_tasks))
            if resolved_preprocessing_tasks is not None
            else count_preprocessing_tasks(
                self.training_config.preprocessing_ids,
                include_combinations=self.training_config.include_combinations,
                param_grids=self.training_config.preprocessing_grids,
            )
        )
        total_experiments = preprocessing_count * len(model_names) * augmentation_count
        folds = self.training_config.folds if self.training_config.folds > 0 else 1
        return ExperimentGridCounts(
            preprocessing_count=preprocessing_count,
            model_count=len(model_names),
            augmentation_count=augmentation_count,
            total_experiments=total_experiments,
            total_fits=total_experiments * folds,
            folds=folds,
        )

    def _iter_queue_entries(
        self,
        *,
        resolved_preprocessing_tasks: Iterable[Any] | None = None,
    ) -> Iterable[tuple[dict[str, Any], TrainingTask]]:
        for training_task in iter_training_tasks(
            self.training_config,
            model_builders=self.model_builders,
            preprocessing_tasks=resolved_preprocessing_tasks,
        ):
            yield build_experiment_record(training_task, self.training_config), training_task

    def _log_queue_estimate(
        self,
        *,
        counts: ExperimentGridCounts,
        max_queue_tasks: int | None,
        queue_export_path: str | Path | None,
    ) -> None:
        message = "Estimated experiment queue size: %s task(s) " "(%s preprocessing variants x %s model(s) x %s augmentation value(s)). " "Estimated fits: %s (folds=%s)."
        self.logger.info(
            message,
            f"{counts.total_experiments:,}",
            f"{counts.preprocessing_count:,}",
            f"{counts.model_count:,}",
            f"{counts.augmentation_count:,}",
            f"{counts.total_fits:,}",
            f"{counts.folds:,}",
        )
        if counts.total_experiments >= HUGE_QUEUE_WARNING_TASKS:
            warning = "Huge queue estimate detected: %s task(s). " "Use `run_experiments.py count` for planning, and explicitly opt in " "with `--allow-huge-queue` for execution."
            self.logger.warning(warning, f"{counts.total_experiments:,}")
            print(warning % f"{counts.total_experiments:,}")
        if max_queue_tasks is None and queue_export_path is None and counts.total_experiments > MAX_QUEUE_TASKS:
            self.logger.warning(
                "Huge queue opt-in is active without queue export. " "State persistence may still be expensive for %s task(s).",
                f"{counts.total_experiments:,}",
            )

    def build_queue_with_options(
        self,
        *,
        max_queue_tasks: int | None = MAX_QUEUE_TASKS,
        queue_export_path: str | Path | None = None,
    ) -> list[tuple[dict[str, Any], TrainingTask]]:
        resolved_preprocessing_tasks = list(self.preprocessing_tasks) if self.preprocessing_tasks is not None else None
        counts = self.estimate_grid_counts(
            resolved_preprocessing_tasks=resolved_preprocessing_tasks,
        )
        self._log_queue_estimate(
            counts=counts,
            max_queue_tasks=max_queue_tasks,
            queue_export_path=queue_export_path,
        )
        if max_queue_tasks is not None and max_queue_tasks <= 0:
            raise ValueError("max_queue_tasks must be > 0 when provided.")
        if max_queue_tasks is not None and counts.total_experiments > max_queue_tasks:
            raise ValueError(
                "The requested experiment grid expands to "
                f"{counts.total_experiments:,} training tasks, which exceeds the "
                f"safety limit of {max_queue_tasks:,}. "
                f"Dimensions: preprocessing={counts.preprocessing_count:,}, "
                f"models={counts.model_count:,}, augmentations={counts.augmentation_count:,}. "
                "Use `--allow-huge-queue` to explicitly opt in."
            )

        if counts.total_experiments > MATERIALIZED_QUEUE_HARD_LIMIT and max_queue_tasks is not None:
            if queue_export_path is not None:
                self._write_queue_export(
                    Path(queue_export_path),
                    self._iter_queue_entries(resolved_preprocessing_tasks=resolved_preprocessing_tasks),
                    task_count=counts.total_experiments,
                )
            export_suffix = f" A queue export was written to {queue_export_path}." if queue_export_path is not None else ""
            raise ValueError(
                "The requested experiment grid expands to "
                f"{counts.total_experiments:,} training tasks, which exceeds the "
                f"in-memory runner safety limit of {MATERIALIZED_QUEUE_HARD_LIMIT:,}. "
                "This branch does not stream persisted runner state for queues of "
                "that size yet. Use `run_experiments.py count` and split execution "
                "in chunks (for example via queue export) when operating at very large scale."
                f"{export_suffix}"
            )

        if resolved_preprocessing_tasks is None:
            queue_entries = [
                (build_experiment_record(training_task, self.training_config), training_task)
                for training_task in build_training_tasks(
                    self.training_config,
                    model_builders=self.model_builders,
                )
            ]
        else:
            queue_entries = list(
                self._iter_queue_entries(
                    resolved_preprocessing_tasks=resolved_preprocessing_tasks,
                )
            )
        if queue_export_path is not None:
            self._write_queue_export(
                Path(queue_export_path),
                queue_entries,
                task_count=len(queue_entries),
            )
        return queue_entries

    def _write_queue_export(
        self,
        queue_export_path: Path,
        queue_entries: Iterable[tuple[dict[str, Any], TrainingTask]],
        *,
        task_count: int,
    ) -> None:
        queue_export_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=queue_export_path.parent,
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "metadata",
                        "generated_at": _timestamp_now(),
                        "config_path": str(self.config_path),
                        "task_count": task_count,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            for record, _training_task in queue_entries:
                line_payload = {"kind": "task", **record}
                handle.write(json.dumps(_normalize_json_value(line_payload), sort_keys=True) + "\n")
            temp_path = Path(handle.name)
        temp_path.replace(queue_export_path)

    def _install_signal_handlers(self) -> None:
        def _handle_signal(signum, frame):  # type: ignore[unused-argument]
            self.store.request_stop(reason=f"signal:{signum}")
            self.logger.info(
                "Received signal %s. The runner will stop after the current task.",
                signum,
            )

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)

    def _restore_signal_handlers(self) -> None:
        for signum, previous_handler in self._previous_signal_handlers.items():
            signal.signal(signum, previous_handler)
        self._previous_signal_handlers.clear()

    def _build_queue_and_sync_state(
        self,
        *,
        max_queue_tasks: int | None = MAX_QUEUE_TASKS,
        queue_export_path: str | Path | None = None,
    ) -> tuple[list[tuple[dict[str, Any], TrainingTask]], dict[str, Any]]:
        queue_entries = self.build_queue_with_options(
            max_queue_tasks=max_queue_tasks,
            queue_export_path=queue_export_path,
        )
        state = self.store.sync_queue([record for record, _ in queue_entries], config_path=self.config_path)
        return queue_entries, state

    def _find_task_in_queue(
        self,
        queue_entries: list[tuple[dict[str, Any], TrainingTask]],
        task_id: str,
    ) -> tuple[dict[str, Any], TrainingTask] | None:
        for record, training_task in queue_entries:
            if record["id"] == task_id:
                return record, training_task
        return None

    def _launch_skip_reason_counts_default(self) -> dict[str, int]:
        return {
            "already_completed": 0,
            "failed_without_rerun": 0,
            "limit_excluded": 0,
        }

    def _build_launch_context(
        self,
        *,
        options: IterativeRunOptions,
        use_stream_queue_mode: bool,
        estimated_total_experiments: int,
        state: dict[str, Any],
        runnable_ids_before_limit: list[str],
        runnable_ids_after_limit: list[str],
    ) -> dict[str, Any]:
        now = _timestamp_now()
        launch_id_seed = f"{now}|pid={os.getpid()}|config={self.config_path}"
        launch_id = f"launch-{hashlib.sha1(launch_id_seed.encode('utf-8')).hexdigest()[:12]}"
        skip_reason_counts = self._launch_skip_reason_counts_default()

        if use_stream_queue_mode:
            planned_runnable_count = int(estimated_total_experiments)
            if options.limit is not None:
                planned_runnable_count = min(planned_runnable_count, int(options.limit))
                skip_reason_counts["limit_excluded"] = max(0, int(estimated_total_experiments) - planned_runnable_count)
        else:
            runnable_after_limit_set = set(runnable_ids_after_limit)
            if options.limit is not None:
                skip_reason_counts["limit_excluded"] = max(0, len(runnable_ids_before_limit) - len(runnable_ids_after_limit))

            for task in state.get("tasks", []):
                task_id = str(task.get("id"))
                status = str(task.get("status", "pending"))
                if task_id in runnable_after_limit_set:
                    continue
                if status == "completed" and not options.rerun_completed:
                    skip_reason_counts["already_completed"] += 1
                    continue
                if status == "failed" and not options.rerun_failed:
                    skip_reason_counts["failed_without_rerun"] += 1
            planned_runnable_count = len(runnable_ids_after_limit)

        skipped_count = int(sum(skip_reason_counts.values()))
        return {
            "launch_id": launch_id,
            "started_at": now,
            "stream_queue_mode": bool(use_stream_queue_mode),
            "rerun_failed": bool(options.rerun_failed),
            "rerun_completed": bool(options.rerun_completed),
            "limit": options.limit,
            "planned_runnable_count": int(planned_runnable_count),
            "skipped_count": skipped_count,
            "skip_reason_counts": skip_reason_counts,
        }

    def _resolve_gpu_visible_devices_for_policy(self, *, device_policy: str) -> str | None:
        if device_policy == "cpu-only":
            return "-1"
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices == "-1":
            self.logger.warning(
                "Ignoring inherited CUDA_VISIBLE_DEVICES=-1 because policy=%s requires GPU attempts.",
                device_policy,
            )
            return None
        return visible_devices

    def _validate_gpu_only_policy(self, runtime_state: dict[str, Any]) -> None:
        env = self._resolve_device_attempt_env(
            requested_device="gpu",
            gpu_visible_devices=runtime_state.get("gpu_visible_devices"),
        )
        probe_result = self._run_device_runtime_probe(
            record=None,
            attempt_number=0,
            requested_device="gpu",
            env=env,
        )
        if not probe_result.get("ok", False):
            errors = [str(item) for item in probe_result.get("errors", [])]
            details = "; ".join(errors) if errors else "unknown GPU runtime probe failure"
            raise RuntimeError("device-policy gpu-only requires a visible and usable GPU, " f"but runtime probe failed: {details}")

    def _task_state_by_id(self, task_id: str) -> dict[str, Any]:
        state = self.store.load_state()
        for task in state["tasks"]:
            if task.get("id") == task_id:
                return task
        raise KeyError(f"Task id not found in runner state: {task_id}")

    def _default_run_task_command_base(
        self,
        *,
        task_cooldown_seconds: float,
        max_queue_tasks: int | None,
        queue_export_path: str | None,
    ) -> tuple[str, ...]:
        command: list[str] = [
            sys.executable,
            str(_project_root() / "run_experiments.py"),
            "run-task",
            "--config",
            str(self.config_path),
            "--raw-data-dir",
            str(self.project_paths.raw_data_dir),
            "--artifacts-dir",
            str(self.project_paths.artifacts_dir),
            "--train-split",
            str(self.training_config.train_split_path),
            "--test-split",
            str(self.training_config.test_split_path),
            "--history-dir",
            str(self.training_config.history_dir),
            "--predictions-dir",
            str(self.training_config.predictions_dir),
            "--folds",
            str(self.training_config.folds),
            "--validation-size",
            str(self.training_config.validation_size),
            "--random-state",
            str(self.training_config.random_state),
            "--batch-size",
            str(self.training_config.batch_size),
            "--epochs",
            str(self.training_config.epochs),
            "--learning-rate",
            str(self.training_config.learning_rate),
            "--loss",
            str(self.training_config.loss),
            "--task-cooldown-seconds",
            str(task_cooldown_seconds),
        ]
        if self.training_config.model_names:
            command.append("--models")
            command.extend(str(name) for name in self.training_config.model_names)
        if self.training_config.preprocessing_ids:
            command.append("--preprocessing")
            command.extend(str(name) for name in self.training_config.preprocessing_ids)
        if self.training_config.augmentation_values:
            command.append("--augmentations-per-image")
            command.extend(str(value) for value in self.training_config.augmentation_values)
        command.append("--combined-preprocessing" if self.training_config.include_combinations else "--no-combined-preprocessing")
        command.append("--run-skip" if self.training_config.run_skip else "--no-run-skip")
        command.extend(self.cpu_execution_limits.to_cli_args())
        if max_queue_tasks is None:
            command.append("--allow-huge-queue")
        else:
            command.extend(["--max-queue-tasks", str(max_queue_tasks)])
        if queue_export_path:
            command.extend(["--queue-export-path", str(queue_export_path)])
        return tuple(command)

    def _run_task_subprocess_command(
        self,
        task_id: str,
        command_base: tuple[str, ...] | None,
        *,
        task_record: dict[str, Any] | None = None,
    ) -> list[str]:
        if command_base is None:
            command = [sys.executable, "run_experiments.py", "run-task"]
        else:
            command = [str(entry) for entry in command_base]
        resolved = [*command, "--task-id", task_id]
        if task_record is not None:
            resolved.extend(
                [
                    "--task-record-json",
                    json.dumps(_normalize_json_value(task_record), sort_keys=True, separators=(",", ":")),
                ]
            )
        return resolved

    def _probe_runtime_subprocess_command(self, requested_device: str) -> list[str]:
        command = [
            sys.executable,
            str(_project_root() / "run_experiments.py"),
            "probe-runtime",
            "--device",
            requested_device,
            "--json",
        ]
        if requested_device == "cpu":
            command.extend(self.cpu_execution_limits.to_cli_args())
        return command

    def _resolve_device_attempt_env(
        self,
        *,
        requested_device: str,
        gpu_visible_devices: str | None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env[ISOLATED_TASK_CHILD_MODE_ENV] = "1"
        env[ISOLATED_TASK_PARENT_PID_ENV] = str(os.getpid())
        env[REQUESTED_DEVICE_ENV] = requested_device
        if requested_device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            env.update(resolve_cpu_thread_env(self.cpu_execution_limits))
        else:
            if gpu_visible_devices is None:
                env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = gpu_visible_devices
        return env

    @staticmethod
    def _probe_failure_kind_for_device(
        *,
        requested_device: str,
        errors: list[str] | tuple[str, ...],
    ) -> str:
        if requested_device != "gpu":
            return "runtime_probe_failed"
        combined = " ".join(str(item).lower() for item in errors)
        unavailable_markers = (
            "no tensorflow gpu devices are visible",
            "could not run a tiny operation on gpu",
            "tiny tensorflow gpu op",
            "tensorflow import failed",
            "cuda_visible_devices",
            "gpu is not available",
        )
        if any(marker in combined for marker in unavailable_markers):
            return "gpu_unavailable"
        return "gpu_probe_failed"

    def _run_device_runtime_probe(
        self,
        *,
        record: dict[str, Any] | None,
        attempt_number: int,
        requested_device: str,
        env: dict[str, str],
    ) -> dict[str, Any]:
        task_id = "runtime-validation" if record is None else record.get("id")
        self._log_structured_phase(
            phase="task:device_attempt_started",
            event="device_attempt_started",
            task_record=record,
            attempt=attempt_number,
            message=f"Starting {requested_device} device attempt for {task_id}.",
            extra={
                "requested_device": requested_device,
            },
        )
        self._log_structured_phase(
            phase="task:device_env_resolved",
            event="device_env_resolved",
            task_record=record,
            attempt=attempt_number,
            extra={
                "requested_device": requested_device,
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "cpu_thread_env": {key: env.get(key) for key in (*resolve_cpu_thread_env(self.cpu_execution_limits),)} if requested_device == "cpu" else current_cpu_thread_env(),
            },
        )

        if requested_device == "gpu":
            self._log_structured_phase(
                phase="task:gpu_probe_started",
                event="gpu_probe_started",
                task_record=record,
                attempt=attempt_number,
                message=f"Running GPU probe before task {task_id}.",
            )

        command = self._probe_runtime_subprocess_command(requested_device)
        try:
            completed = subprocess.run(
                command,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
        except subprocess.TimeoutExpired as error:
            stdout = (error.stdout or "").strip()
            stderr = (error.stderr or "").strip()
            completed = None

        payload: dict[str, Any] | None = None
        payload_error: str | None = None
        if stdout:
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as error:
                payload_error = f"Could not parse runtime probe JSON output: {error}"
        elif completed is None:
            payload_error = "Runtime probe subprocess timed out."

        probe_errors: list[str] = []
        probe_warnings: list[str] = []
        if payload is not None:
            probe_errors.extend(str(item) for item in payload.get("errors", []))
            probe_warnings.extend(str(item) for item in payload.get("warnings", []))
        if payload_error:
            probe_errors.append(payload_error)
        if stderr:
            probe_warnings.append(stderr[:2_000])
        if completed is not None and int(completed.returncode) != 0 and not probe_errors:
            probe_errors.append(f"Runtime probe exited with code {int(completed.returncode)}.")

        result = {
            "ok": bool(payload is not None and payload.get("ok") and not probe_errors and (completed is None or int(completed.returncode) == 0)),
            "requested_device": requested_device,
            "effective_device": ("cpu" if requested_device == "cpu" else str((payload or {}).get("effective_device") or "cpu")),
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "tensorflow_visible_devices": list((payload or {}).get("tensorflow_visible_devices", [])),
            "gpu_used": bool((payload or {}).get("gpu_used", False)),
            "effective_cpu_thread_env": dict((payload or {}).get("effective_cpu_thread_env", {})),
            "errors": probe_errors,
            "warnings": probe_warnings,
            "payload": payload or {},
            "returncode": None if completed is None else int(completed.returncode),
        }

        if requested_device == "gpu":
            event_name = "gpu_probe_succeeded" if result["ok"] else "gpu_probe_failed"
            self._log_structured_phase(
                phase=f"task:{event_name}",
                event=event_name,
                task_record=record,
                attempt=attempt_number,
                message=(f"GPU probe succeeded for {task_id}." if result["ok"] else f"GPU probe failed for {task_id}."),
                extra={
                    "requested_device": requested_device,
                    "effective_device": result["effective_device"],
                    "cuda_visible_devices": result["cuda_visible_devices"],
                    "tensorflow_visible_devices": result["tensorflow_visible_devices"],
                    "gpu_used": result["gpu_used"],
                    "probe_errors": result["errors"],
                    "probe_warnings": result["warnings"],
                },
            )

        return result

    def _build_failed_probe_attempt_result(
        self,
        *,
        record: dict[str, Any],
        attempt_number: int,
        probe_result: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = record["id"]
        requested_device = str(probe_result.get("requested_device", "unknown"))
        effective_device = str(probe_result.get("effective_device", requested_device))
        probe_errors = [str(item) for item in probe_result.get("errors", [])]
        failure_kind = self._probe_failure_kind_for_device(
            requested_device=requested_device,
            errors=probe_errors,
        )
        error_summary = "; ".join(probe_errors)[:500]
        self.store.update_task_status(
            task_id,
            status="failed",
            error_summary=error_summary or f"{failure_kind} prevented task launch.",
            duration_seconds=0.0,
        )
        self.store.append_task_attempt(
            task_id,
            {
                "task_id": task_id,
                "attempt": attempt_number,
                "device": requested_device,
                "requested_device": requested_device,
                "effective_device": effective_device,
                "cuda_visible_devices": probe_result.get("cuda_visible_devices"),
                "tensorflow_visible_devices": probe_result.get("tensorflow_visible_devices", []),
                "gpu_used": bool(probe_result.get("gpu_used", False)),
                "effective_cpu_thread_env": probe_result.get("effective_cpu_thread_env", {}),
                "exit_code": probe_result.get("returncode"),
                "failure_kind": failure_kind,
                "started_at": _timestamp_now(),
                "finished_at": _timestamp_now(),
                "duration_seconds": 0.0,
            },
        )
        self.store.update_task_execution_details(
            task_id,
            final_device=effective_device,
            failure_kind=failure_kind,
        )
        return {
            "task_id": task_id,
            "status": "failed",
            "duration_seconds": 0.0,
            "exit_code": probe_result.get("returncode"),
            "timeout": False,
            "failure_kind": failure_kind,
            "device": requested_device,
            "requested_device": requested_device,
            "effective_device": effective_device,
            "cuda_visible_devices": probe_result.get("cuda_visible_devices"),
            "tensorflow_visible_devices": probe_result.get("tensorflow_visible_devices", []),
            "gpu_used": bool(probe_result.get("gpu_used", False)),
            "effective_cpu_thread_env": probe_result.get("effective_cpu_thread_env", {}),
            "attempt": attempt_number,
            "error_summary": error_summary or f"{failure_kind} prevented task launch.",
        }

    def _execute_task_entry_isolated_subprocess(
        self,
        *,
        record: dict[str, Any],
        command_base: tuple[str, ...] | None,
        timeout_seconds: float | None,
        attempt_number: int,
        requested_device: str,
        env: dict[str, str],
        probe_result: dict[str, Any],
        task_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_id = record["id"]
        command = self._run_task_subprocess_command(task_id, command_base, task_record=task_record)
        started_at = datetime.now(timezone.utc)
        self.logger.info(
            "Running isolated task %s via subprocess (attempt=%s, device=%s): %s",
            task_id,
            attempt_number,
            requested_device,
            " ".join(command),
        )
        print(f"Running isolated task {task_id} (attempt={attempt_number}, device={requested_device})")
        self._log_structured_phase(
            phase="task:subprocess_start",
            task_record=record,
            message=f"Launching isolated subprocess for {task_id}.",
            attempt=attempt_number,
            extra={
                "subprocess_command": command,
                "task_timeout_seconds": timeout_seconds,
                "device": requested_device,
                "requested_device": requested_device,
                "effective_device": probe_result.get("effective_device"),
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "tensorflow_visible_devices": probe_result.get("tensorflow_visible_devices", []),
                "gpu_used": probe_result.get("gpu_used", False),
            },
        )

        try:
            completed = subprocess.run(
                command,
                env=env,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
            failure_kind = classify_task_failure(
                device=requested_device,
                exit_code=None,
                timeout=True,
                error_summary="subprocess timeout",
            )
            error_summary = "Isolated task subprocess timed out after " f"{timeout_seconds:.2f}s."
            self.store.update_task_status(
                task_id,
                status="failed",
                error_summary=error_summary[:500],
                duration_seconds=duration_seconds,
            )
            finished_at = _timestamp_now()
            self.store.append_task_attempt(
                task_id,
                {
                    "task_id": task_id,
                    "attempt": attempt_number,
                    "device": requested_device,
                    "requested_device": requested_device,
                    "effective_device": probe_result.get("effective_device", requested_device),
                    "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                    "tensorflow_visible_devices": probe_result.get("tensorflow_visible_devices", []),
                    "gpu_used": probe_result.get("gpu_used", False),
                    "effective_cpu_thread_env": probe_result.get("effective_cpu_thread_env", {}),
                    "exit_code": None,
                    "failure_kind": failure_kind,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at,
                    "duration_seconds": duration_seconds,
                },
            )
            self.store.update_task_execution_details(
                task_id,
                final_device=str(probe_result.get("effective_device", requested_device)),
                failure_kind=failure_kind,
            )
            self.logger.error(
                "Isolated task %s timed out after %.2fs.",
                task_id,
                duration_seconds,
            )
            print(f"Task {task_id} timed out after {timeout_seconds:.2f}s")
            self._log_structured_phase(
                phase="task:timeout",
                task_record=record,
                attempt=attempt_number,
                message=error_summary,
                extra={
                    "duration_seconds": duration_seconds,
                    "subprocess_command": command,
                    "task_timeout_seconds": timeout_seconds,
                    "device": requested_device,
                    "failure_kind": failure_kind,
                },
            )
            return {
                "task_id": task_id,
                "status": "failed",
                "duration_seconds": duration_seconds,
                "timeout": True,
                "exit_code": None,
                "error_summary": error_summary,
                "failure_kind": failure_kind,
                "device": requested_device,
                "requested_device": requested_device,
                "effective_device": str(probe_result.get("effective_device", requested_device)),
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "tensorflow_visible_devices": probe_result.get("tensorflow_visible_devices", []),
                "gpu_used": bool(probe_result.get("gpu_used", False)),
                "effective_cpu_thread_env": probe_result.get("effective_cpu_thread_env", {}),
                "attempt": attempt_number,
            }

        duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
        exit_code = int(completed.returncode)
        task_state = self._task_state_by_id(task_id)
        task_status = str(task_state.get("status", "pending"))
        task_events_path, _task_memory_path, task_log_path = self._task_log_paths(task_id)
        error_summary = task_state.get("error_summary")
        failure_kind = classify_task_failure(
            device=requested_device,
            exit_code=exit_code,
            timeout=False,
            error_summary=None if error_summary is None else str(error_summary),
            task_events_path=task_events_path,
            task_log_path=task_log_path,
        )

        if exit_code != 0 and task_status not in {"failed", "stopped"}:
            if task_status not in {"completed"}:
                error_summary = f"Isolated task subprocess exited with code {exit_code}."
                self.store.update_task_status(
                    task_id,
                    status="failed",
                    error_summary=error_summary,
                    duration_seconds=duration_seconds,
                )
                task_status = "failed"
                task_state = self._task_state_by_id(task_id)
        elif exit_code == 0 and task_status in {"pending", "running"}:
            error_summary = "Isolated task subprocess exited successfully, but task status " f"remained '{task_status}'."
            self.store.update_task_status(
                task_id,
                status="failed",
                error_summary=error_summary[:500],
                duration_seconds=duration_seconds,
            )
            task_status = "failed"
            task_state = self._task_state_by_id(task_id)
            error_summary = task_state.get("error_summary")

        finished_at = _timestamp_now()
        self.store.append_task_attempt(
            task_id,
            {
                "task_id": task_id,
                "attempt": attempt_number,
                "device": requested_device,
                "requested_device": requested_device,
                "effective_device": probe_result.get("effective_device", requested_device),
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "tensorflow_visible_devices": probe_result.get("tensorflow_visible_devices", []),
                "gpu_used": probe_result.get("gpu_used", False),
                "effective_cpu_thread_env": probe_result.get("effective_cpu_thread_env", {}),
                "exit_code": exit_code,
                "failure_kind": (None if task_status == "completed" else failure_kind),
                "started_at": started_at.isoformat(),
                "finished_at": finished_at,
                "duration_seconds": duration_seconds,
            },
        )
        if task_status == "completed":
            self.store.update_task_execution_details(
                task_id,
                final_device=str(probe_result.get("effective_device", requested_device)),
                failure_kind=None,
            )
        else:
            self.store.update_task_execution_details(
                task_id,
                final_device=str(probe_result.get("effective_device", requested_device)),
                failure_kind=failure_kind,
            )

        self.logger.info(
            "Isolated task %s finished with exit code %s and status %s.",
            task_id,
            exit_code,
            task_status,
        )
        self._log_structured_phase(
            phase="task:subprocess_exit",
            task_record=record,
            attempt=attempt_number,
            message=f"Isolated subprocess finished with exit code {exit_code}.",
            extra={
                "duration_seconds": duration_seconds,
                "subprocess_exit_code": exit_code,
                "task_status": task_status,
                "subprocess_command": command,
                "task_error_summary": task_state.get("error_summary"),
                "device": requested_device,
                "requested_device": requested_device,
                "effective_device": probe_result.get("effective_device"),
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "tensorflow_visible_devices": probe_result.get("tensorflow_visible_devices", []),
                "gpu_used": probe_result.get("gpu_used", False),
                "failure_kind": (None if task_status == "completed" else failure_kind),
            },
        )
        if exit_code == 0:
            print(f"Task {task_id} finished with status={task_status} (device={requested_device})")
        else:
            print(f"Task {task_id} subprocess failed with exit code {exit_code} (device={requested_device})")
        return {
            "task_id": task_id,
            "status": task_status,
            "duration_seconds": duration_seconds,
            "exit_code": exit_code,
            "timeout": False,
            "failure_kind": (None if task_status == "completed" else failure_kind),
            "device": requested_device,
            "requested_device": requested_device,
            "effective_device": str(probe_result.get("effective_device", requested_device)),
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "tensorflow_visible_devices": probe_result.get("tensorflow_visible_devices", []),
            "gpu_used": bool(probe_result.get("gpu_used", False)),
            "effective_cpu_thread_env": probe_result.get("effective_cpu_thread_env", {}),
            "attempt": attempt_number,
        }

    def _apply_cpu_limits_for_current_process(
        self,
        *,
        record: dict[str, Any] | None = None,
        attempt: int | None = None,
    ) -> dict[str, Any] | None:
        requested_device = os.environ.get(REQUESTED_DEVICE_ENV)
        cpu_requested = requested_device == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        if not cpu_requested or self._cpu_limits_applied:
            return None

        applied = apply_cpu_runtime_limits(self.cpu_execution_limits)
        applied["requested_device"] = requested_device or "cpu"
        applied["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        self._cpu_limits_applied = True
        if record is not None:
            self._log_structured_phase(
                phase="task:cpu_limits_applied",
                event="cpu_limits_applied",
                task_record=record,
                attempt=attempt,
                message=f"Applied CPU runtime limits for {record['id']}.",
                extra=applied,
            )
        return applied

    def _is_isolated_child_mode(self) -> bool:
        return os.environ.get(ISOLATED_TASK_CHILD_MODE_ENV) == "1"

    def _sleep_with_log(self, seconds: float, *, reason: str) -> None:
        delay = max(0.0, float(seconds))
        if delay <= 0:
            return
        self.logger.info("Sleeping for %.2fs (%s).", delay, reason)
        time.sleep(delay)

    @staticmethod
    def _is_hot_metric(value: float | None, limit: float | None) -> bool:
        if value is None or limit is None:
            return False
        return value >= limit

    def _apply_thermal_policy_before_task(
        self,
        *,
        record: dict[str, Any],
        options: IterativeRunOptions,
        runtime_state: dict[str, Any],
    ) -> tuple[str | None, float]:
        if not bool(options.thermal_policy_enabled):
            runtime_state["thermal_policy_enabled"] = False
            runtime_state["thermal_state"] = "disabled"
            runtime_state["thermal_last_reason"] = None
            return None, 0.0

        snapshot = collect_thermal_snapshot()
        runtime_state["thermal_policy_enabled"] = True
        runtime_state["thermal_last_sample_at"] = _timestamp_now()
        runtime_state["thermal_last_cpu_temp_celsius"] = snapshot.cpu_temperature_celsius
        runtime_state["thermal_last_cpu_load_percent"] = snapshot.cpu_load_percent
        runtime_state["thermal_last_gpu_temp_celsius"] = snapshot.max_gpu_temperature_celsius
        runtime_state["thermal_last_gpu_utilization_percent"] = snapshot.max_gpu_utilization_percent

        cpu_hot = self._is_hot_metric(
            snapshot.cpu_temperature_celsius,
            options.thermal_cpu_temp_celsius_limit,
        ) or self._is_hot_metric(
            snapshot.cpu_load_percent,
            options.thermal_cpu_load_percent_limit,
        )
        gpu_hot = self._is_hot_metric(
            snapshot.max_gpu_temperature_celsius,
            options.thermal_gpu_temp_celsius_limit,
        ) or self._is_hot_metric(
            snapshot.max_gpu_utilization_percent,
            options.thermal_gpu_utilization_percent_limit,
        )
        recovery_temp = options.thermal_gpu_recovery_temp_celsius
        if recovery_temp is not None and snapshot.max_gpu_temperature_celsius is not None:
            gpu_hot = gpu_hot or snapshot.max_gpu_temperature_celsius > recovery_temp

        warnings = list(snapshot.warnings)
        if warnings:
            self._log_structured_phase(
                phase="task:thermal_probe_warning",
                event="thermal_probe_warning",
                task_record=record,
                message=f"Thermal probe warning(s) for {record['id']}.",
                extra={"warnings": warnings},
            )

        cooldown_seconds = max(0.0, float(options.thermal_cooldown_seconds))
        now = datetime.now(timezone.utc)
        forced_device: str | None = None
        thermal_state = "normal"
        reason = "thermal metrics are below configured limits."

        if cpu_hot and gpu_hot:
            thermal_state = "both_hot"
            reason = "CPU and GPU are above configured thermal limits."
            if options.device_policy in {"adaptive", "gpu-first"}:
                forced_device = "cpu"
            self._log_structured_phase(
                phase="task:thermal_both_hot_pause",
                event="thermal_both_hot_pause",
                task_record=record,
                message=f"Pausing before task {record['id']} because CPU and GPU are hot.",
                extra={
                    "cooldown_seconds": cooldown_seconds,
                    "cpu_temperature_celsius": snapshot.cpu_temperature_celsius,
                    "cpu_load_percent": snapshot.cpu_load_percent,
                    "gpu_temperature_celsius": snapshot.max_gpu_temperature_celsius,
                    "gpu_utilization_percent": snapshot.max_gpu_utilization_percent,
                },
            )
        elif gpu_hot:
            thermal_state = "gpu_hot"
            reason = "GPU is above configured thermal limits."
            if options.device_policy in {"adaptive", "gpu-first"}:
                forced_device = "cpu"
            self._log_structured_phase(
                phase="task:thermal_gpu_hot",
                event="thermal_gpu_hot",
                task_record=record,
                message=f"GPU thermal limit reached before task {record['id']}.",
                extra={
                    "cpu_temperature_celsius": snapshot.cpu_temperature_celsius,
                    "cpu_load_percent": snapshot.cpu_load_percent,
                    "gpu_temperature_celsius": snapshot.max_gpu_temperature_celsius,
                    "gpu_utilization_percent": snapshot.max_gpu_utilization_percent,
                    "forced_device": forced_device,
                },
            )
        elif cpu_hot:
            thermal_state = "cpu_hot"
            reason = "CPU is above configured thermal limits."
            self._log_structured_phase(
                phase="task:thermal_cpu_hot_pause",
                event="thermal_cpu_hot_pause",
                task_record=record,
                message=f"Pausing before task {record['id']} because CPU is hot.",
                extra={
                    "cooldown_seconds": cooldown_seconds,
                    "cpu_temperature_celsius": snapshot.cpu_temperature_celsius,
                    "cpu_load_percent": snapshot.cpu_load_percent,
                    "gpu_temperature_celsius": snapshot.max_gpu_temperature_celsius,
                    "gpu_utilization_percent": snapshot.max_gpu_utilization_percent,
                },
            )
        elif runtime_state.get("thermal_state") in {"gpu_hot", "both_hot"} and runtime_state.get("preferred_device") == "cpu":
            self._log_structured_phase(
                phase="task:thermal_recovered",
                event="thermal_recovered",
                task_record=record,
                message=f"Thermal conditions recovered before task {record['id']}.",
                extra={
                    "cpu_temperature_celsius": snapshot.cpu_temperature_celsius,
                    "cpu_load_percent": snapshot.cpu_load_percent,
                    "gpu_temperature_celsius": snapshot.max_gpu_temperature_celsius,
                    "gpu_utilization_percent": snapshot.max_gpu_utilization_percent,
                },
            )

        runtime_state["thermal_state"] = thermal_state
        runtime_state["thermal_last_reason"] = reason
        if thermal_state in {"gpu_hot", "both_hot"}:
            runtime_state["preferred_device"] = "cpu"
            runtime_state["gpu_health"] = "cooling_down"
            runtime_state["gpu_recovery_cooldown_until"] = datetime.fromtimestamp(
                now.timestamp() + cooldown_seconds,
                tz=timezone.utc,
            ).isoformat()
        if thermal_state in {"cpu_hot", "both_hot"} and cooldown_seconds > 0:
            pause_reason = "thermal_both_hot" if thermal_state == "both_hot" else "thermal_cpu_hot"
            self._sleep_with_log(cooldown_seconds, reason=pause_reason)

        return forced_device, cooldown_seconds

    def _task_record_for_id(
        self,
        queue_entries: list[tuple[dict[str, Any], TrainingTask]],
        task_id: str,
    ) -> dict[str, Any]:
        for record, _ in queue_entries:
            if record["id"] == task_id:
                return record
        raise KeyError(f"Task id not found in queue: {task_id}")

    def _run_task_with_isolated_device_policy(
        self,
        *,
        record: dict[str, Any],
        options: IterativeRunOptions,
        runtime_state: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = record["id"]
        policy = options.device_policy
        run_task_command_base = (
            options.run_task_command_base
            if options.run_task_command_base is not None
            else self._default_run_task_command_base(
                task_cooldown_seconds=self._task_cooldown_seconds,
                max_queue_tasks=options.max_queue_tasks,
                queue_export_path=options.queue_export_path,
            )
        )
        attempt_results: list[dict[str, Any]] = []
        attempted_devices: list[str] = []
        gpu_retries_remaining = int(options.gpu_retries)
        cpu_retries_remaining = int(options.cpu_retries)
        total_attempts = 0
        next_device = "cpu" if policy == "cpu-only" else str(runtime_state.get("preferred_device", "gpu"))
        self.store.register_task_record(record, config_path=self.config_path)

        if policy == "gpu-only":
            next_device = "gpu"
        if policy == "gpu-first":
            next_device = "gpu"
        forced_thermal_device, _thermal_cooldown_seconds = self._apply_thermal_policy_before_task(
            record=record,
            options=options,
            runtime_state=runtime_state,
        )
        thermal_forced_device = forced_thermal_device is not None
        if forced_thermal_device is not None:
            next_device = forced_thermal_device
        if policy == "adaptive":
            cooldown_until = _parse_timestamp(runtime_state.get("gpu_recovery_cooldown_until"))
            if not thermal_forced_device and next_device == "cpu" and cooldown_until is not None and datetime.now(timezone.utc) >= cooldown_until:
                next_device = "gpu"
                runtime_state["preferred_device"] = "gpu"
                runtime_state["gpu_health"] = "cooling_down"
                self._log_structured_phase(
                    phase="task:gpu_recovery_probe_scheduled",
                    event="gpu_recovery_probe_scheduled",
                    task_record=record,
                    message=f"GPU recovery probe scheduled for {task_id}.",
                )

        while total_attempts < int(options.max_task_attempts):
            total_attempts += 1
            attempt_env = self._resolve_device_attempt_env(
                requested_device=next_device,
                gpu_visible_devices=runtime_state.get("gpu_visible_devices"),
            )
            if os.environ.get(TEST_TASK_SHIM_PATH_ENV):
                probe_result = {
                    "ok": True,
                    "requested_device": next_device,
                    "effective_device": ("cpu" if next_device == "cpu" else "gpu"),
                    "cuda_visible_devices": attempt_env.get("CUDA_VISIBLE_DEVICES"),
                    "tensorflow_visible_devices": ([] if next_device == "cpu" else ["/device:GPU:0"]),
                    "gpu_used": next_device == "gpu",
                    "effective_cpu_thread_env": (resolve_cpu_thread_env(self.cpu_execution_limits) if next_device == "cpu" else {}),
                    "errors": [],
                    "warnings": [],
                    "payload": {},
                    "returncode": 0,
                }
                self._log_structured_phase(
                    phase="task:device_attempt_started",
                    event="device_attempt_started",
                    task_record=record,
                    attempt=total_attempts,
                    message=f"Starting {next_device} device attempt for {task_id} (test shim probe bypass).",
                    extra={
                        "requested_device": next_device,
                        "probe_bypassed": True,
                    },
                )
            else:
                probe_result = self._run_device_runtime_probe(
                    record=record,
                    attempt_number=total_attempts,
                    requested_device=next_device,
                    env=attempt_env,
                )
            if not probe_result.get("ok", False):
                attempt_result = self._build_failed_probe_attempt_result(
                    record=record,
                    attempt_number=total_attempts,
                    probe_result=probe_result,
                )
            else:
                attempt_result = self._execute_task_entry_isolated_subprocess(
                    record=record,
                    command_base=run_task_command_base,
                    timeout_seconds=options.task_timeout_seconds,
                    attempt_number=total_attempts,
                    requested_device=next_device,
                    env=attempt_env,
                    probe_result=probe_result,
                    task_record=record,
                )
            attempt_results.append(attempt_result)
            attempted_devices.append(next_device)
            failure_kind = attempt_result.get("failure_kind")
            task_status = str(attempt_result.get("status", "failed"))
            effective_device = str(attempt_result.get("effective_device", next_device))
            fallback_next: str | None = None
            cooldown_seconds = 0.0

            if task_status == "completed":
                runtime_state["last_successful_device"] = next_device
                runtime_state["consecutive_final_oom_failures"] = 0
                if next_device == "gpu":
                    if runtime_state.get("gpu_health") in {"cooling_down", "unhealthy"}:
                        self._log_structured_phase(
                            phase="task:gpu_recovered",
                            event="gpu_recovered",
                            task_record=record,
                            attempt=total_attempts,
                            message=f"GPU recovered on task {task_id}.",
                        )
                    runtime_state["preferred_device"] = "gpu"
                    runtime_state["gpu_health"] = "healthy"
                    runtime_state["gpu_recovery_cooldown_until"] = None
                else:
                    used_gpu_in_this_task = "gpu" in attempted_devices
                    if used_gpu_in_this_task:
                        runtime_state["cpu_fallback_successes"] = int(runtime_state.get("cpu_fallback_successes", 0)) + 1
                        self._log_structured_phase(
                            phase="task:cpu_fallback_succeeded",
                            event="cpu_fallback_succeeded",
                            task_record=record,
                            attempt=total_attempts,
                            message=f"CPU fallback succeeded for {task_id}.",
                        )
                    if policy == "adaptive":
                        runtime_state["preferred_device"] = "cpu"
                        runtime_state["gpu_health"] = "cooling_down"
                        if used_gpu_in_this_task:
                            runtime_state["gpu_recovery_cooldown_until"] = datetime.fromtimestamp(
                                datetime.now(timezone.utc).timestamp() + float(options.gpu_recovery_cooldown_seconds),
                                tz=timezone.utc,
                            ).isoformat()
                        elif runtime_state.get("gpu_recovery_cooldown_until") is None:
                            runtime_state["gpu_recovery_cooldown_until"] = datetime.fromtimestamp(
                                datetime.now(timezone.utc).timestamp() + float(options.gpu_recovery_cooldown_seconds),
                                tz=timezone.utc,
                            ).isoformat()
                self._log_structured_phase(
                    phase="task:attempt_finished",
                    event="task_attempt_finished",
                    task_record=record,
                    attempt=total_attempts,
                    extra={
                        "task_id": task_id,
                        "device": next_device,
                        "effective_device": effective_device,
                        "exit_code": attempt_result.get("exit_code"),
                        "failure_kind": None,
                        "fallback_next": None,
                        "cooldown_seconds": 0,
                    },
                )
                return {
                    **attempt_result,
                    "attempts": attempt_results,
                    "final_device": effective_device,
                    "failure_kind": None,
                }

            is_gpu_probe_failure = failure_kind in {"gpu_probe_failed", "gpu_unavailable"}
            is_gpu_oom = failure_kind == "gpu_oom"
            is_cpu_oom = failure_kind == "cpu_oom"
            is_any_oom = failure_kind in {"oom", "gpu_oom", "cpu_oom"}
            if is_gpu_oom:
                runtime_state["gpu_oom_count"] = int(runtime_state.get("gpu_oom_count", 0)) + 1
                runtime_state["last_gpu_oom_task_id"] = task_id
                self._log_structured_phase(
                    phase="task:gpu_oom_detected",
                    event="gpu_oom_detected",
                    task_record=record,
                    attempt=total_attempts,
                    message=f"GPU OOM detected while running {task_id}.",
                    extra={
                        "task_id": task_id,
                        "failure_kind": failure_kind,
                    },
                )

            if options.fail_fast_on_oom and is_any_oom:
                self._log_structured_phase(
                    phase="task:attempt_finished",
                    event="task_attempt_finished",
                    task_record=record,
                    attempt=total_attempts,
                    extra={
                        "task_id": task_id,
                        "device": next_device,
                        "effective_device": effective_device,
                        "exit_code": attempt_result.get("exit_code"),
                        "failure_kind": failure_kind,
                        "fallback_next": None,
                        "cooldown_seconds": 0,
                    },
                )
                break

            if next_device == "gpu" and is_any_oom and gpu_retries_remaining > 0:
                gpu_retries_remaining -= 1
                fallback_next = "gpu"
                cooldown_seconds = max(0.0, float(options.cooldown_after_oom_seconds))
                self._log_structured_phase(
                    phase="task:gpu_retry_scheduled",
                    event="gpu_retry_scheduled",
                    task_record=record,
                    attempt=total_attempts,
                    message=f"Scheduling GPU retry for {task_id}.",
                    extra={
                        "remaining_gpu_retries": gpu_retries_remaining,
                        "cooldown_seconds": cooldown_seconds,
                    },
                )
                self._log_structured_phase(
                    phase="task:attempt_finished",
                    event="task_attempt_finished",
                    task_record=record,
                    attempt=total_attempts,
                    extra={
                        "task_id": task_id,
                        "device": next_device,
                        "effective_device": effective_device,
                        "exit_code": attempt_result.get("exit_code"),
                        "failure_kind": failure_kind,
                        "fallback_next": fallback_next,
                        "cooldown_seconds": cooldown_seconds,
                    },
                )
                self._sleep_with_log(cooldown_seconds, reason="gpu_oom_retry")
                next_device = "gpu"
                continue

            if next_device == "gpu" and (is_any_oom or is_gpu_probe_failure) and policy in {"gpu-first", "adaptive"}:
                if cpu_retries_remaining >= 0 and total_attempts < int(options.max_task_attempts):
                    fallback_next = "cpu"
                    runtime_state["preferred_device"] = "cpu"
                    runtime_state["gpu_health"] = "unhealthy"
                    cooldown_until = datetime.now(timezone.utc).timestamp() + float(options.gpu_recovery_cooldown_seconds)
                    runtime_state["gpu_recovery_cooldown_until"] = datetime.fromtimestamp(
                        cooldown_until,
                        tz=timezone.utc,
                    ).isoformat()
                    self.store.update_task_execution_details(
                        task_id,
                        fallback_reason=("gpu_unavailable" if is_gpu_probe_failure else "gpu_oom"),
                    )
                    self._log_structured_phase(
                        phase="task:cpu_fallback_scheduled",
                        event="cpu_fallback_scheduled",
                        task_record=record,
                        attempt=total_attempts,
                        message=(f"Scheduling CPU fallback for {task_id} after GPU probe failure." if is_gpu_probe_failure else f"Scheduling CPU fallback for {task_id}."),
                    )
                    self._log_structured_phase(
                        phase="task:attempt_finished",
                        event="task_attempt_finished",
                        task_record=record,
                        attempt=total_attempts,
                        extra={
                            "task_id": task_id,
                            "device": next_device,
                            "effective_device": effective_device,
                            "exit_code": attempt_result.get("exit_code"),
                            "failure_kind": failure_kind,
                            "fallback_next": fallback_next,
                            "cooldown_seconds": 0,
                        },
                    )
                    next_device = "cpu"
                    continue

            if next_device == "cpu" and is_cpu_oom and cpu_retries_remaining > 0:
                cpu_retries_remaining -= 1
                fallback_next = "cpu"
                cooldown_seconds = max(0.0, float(options.cooldown_after_oom_seconds))
                self._log_structured_phase(
                    phase="task:attempt_finished",
                    event="task_attempt_finished",
                    task_record=record,
                    attempt=total_attempts,
                    extra={
                        "task_id": task_id,
                        "device": next_device,
                        "effective_device": effective_device,
                        "exit_code": attempt_result.get("exit_code"),
                        "failure_kind": failure_kind,
                        "fallback_next": fallback_next,
                        "cooldown_seconds": cooldown_seconds,
                    },
                )
                self._sleep_with_log(cooldown_seconds, reason="cpu_oom_retry")
                next_device = "cpu"
                continue

            self._log_structured_phase(
                phase="task:attempt_finished",
                event="task_attempt_finished",
                task_record=record,
                attempt=total_attempts,
                extra={
                    "task_id": task_id,
                    "device": next_device,
                    "effective_device": effective_device,
                    "exit_code": attempt_result.get("exit_code"),
                    "failure_kind": failure_kind,
                    "fallback_next": fallback_next,
                    "cooldown_seconds": cooldown_seconds,
                },
            )
            break

        final_result = attempt_results[-1]
        final_failure_kind = final_result.get("failure_kind")
        is_final_oom = final_failure_kind in {"oom", "gpu_oom", "cpu_oom"}
        if is_final_oom:
            runtime_state["consecutive_final_oom_failures"] = int(runtime_state.get("consecutive_final_oom_failures", 0)) + 1
        else:
            runtime_state["consecutive_final_oom_failures"] = 0
        return {
            **final_result,
            "attempts": attempt_results,
            "final_device": final_result.get("effective_device", final_result.get("device")),
            "failure_kind": final_failure_kind,
        }

    def _execute_task_entry(
        self,
        *,
        record: dict[str, Any],
        training_task: TrainingTask,
        train_df: Any,
        test_df: Any,
    ) -> dict[str, Any]:
        task_id = record["id"]
        self.logger.info("Running %s", training_task.label)
        print(f"Running {training_task.label}")
        updated_state = self.store.update_task_status(task_id, status="running")
        task_snapshot = next(task for task in updated_state["tasks"] if task["id"] == task_id)
        attempt = int(task_snapshot.get("attempts", 1))
        started_at = datetime.now(timezone.utc)
        self._apply_cpu_limits_for_current_process(record=record, attempt=attempt)

        clear_ml_memory()
        if self._task_cooldown_seconds > 0:
            time.sleep(self._task_cooldown_seconds)
        self._log_structured_phase(
            phase="task:start",
            task_record=record,
            attempt=attempt,
            message=f"Running {training_task.label}",
        )

        try:

            def _phase_observer(phase: str, details: dict[str, Any] | None) -> None:
                self._log_structured_phase(
                    phase=phase,
                    task_record=record,
                    attempt=attempt,
                    extra=details,
                )

            result = run_training_task(
                training_task,
                self.training_config,
                train_df=train_df,
                test_df=test_df,
                model_builders=self.model_builders,
                phase_observer=_phase_observer,
            )
            history_summary = build_history_row(result, self.training_config)
            self._log_structured_phase(
                phase="before_save_artifacts",
                task_record=record,
                attempt=attempt,
            )
            history_path, predictions_path = save_run_result(result, self.training_config)
            self._log_structured_phase(
                phase="after_save_artifacts",
                task_record=record,
                attempt=attempt,
                extra={
                    "history_path": str(history_path),
                    "predictions_path": str(predictions_path),
                },
            )
            duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
            self.store.set_task_artifacts(
                task_id,
                history_path=history_path,
                predictions_path=predictions_path,
            )
            self.store.update_task_status(
                task_id,
                status="completed",
                result_summary=history_summary,
                duration_seconds=duration_seconds,
            )
            self.logger.info(
                "Completed %s in %.2fs.",
                training_task.label,
                duration_seconds,
            )
            print(f"Completed {training_task.label} " f"(history: {history_path}, predictions: {predictions_path})")
            result = None
            history_summary = None
            clear_ml_memory()
            self._log_structured_phase(
                phase="after_cleanup",
                task_record=record,
                attempt=attempt,
            )
            self._log_structured_phase(
                phase="task:completed",
                task_record=record,
                attempt=attempt,
                message=f"Completed in {duration_seconds:.2f}s",
                extra={"duration_seconds": duration_seconds},
            )
            return {
                "task_id": task_id,
                "status": "completed",
                "task_label": training_task.label,
                "duration_seconds": duration_seconds,
                "history_path": str(history_path),
                "predictions_path": str(predictions_path),
                "attempt": attempt,
            }
        except Exception as error:
            duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
            error_summary = f"{error.__class__.__name__}: {error}"
            self.store.update_task_status(
                task_id,
                status="failed",
                error_summary=error_summary[:500],
                duration_seconds=duration_seconds,
            )
            self.logger.error(
                "Failed %s in %.2fs: %s\n%s",
                training_task.label,
                duration_seconds,
                error_summary,
                traceback.format_exc(),
            )
            print(f"Failed {training_task.label}: {error_summary}")
            clear_ml_memory()
            traceback_summary = self._traceback_summary(error)
            self._log_structured_phase(
                phase="after_cleanup",
                task_record=record,
                attempt=attempt,
                error=error,
                traceback_summary=traceback_summary,
            )
            self._log_structured_phase(
                phase="task:failed",
                task_record=record,
                attempt=attempt,
                message=f"Failed in {duration_seconds:.2f}s",
                error=error,
                traceback_summary=traceback_summary,
                extra={
                    "duration_seconds": duration_seconds,
                    "task_label": training_task.label,
                    "task_metadata": {
                        "id": record["id"],
                        "model_name": record["model_name"],
                        "preproc_id": record["preproc_id"],
                        "param_id": record["param_id"],
                        "param_display": record["param_display"],
                    },
                },
            )
            return {
                "task_id": task_id,
                "status": "failed",
                "task_label": training_task.label,
                "duration_seconds": duration_seconds,
                "error_summary": error_summary[:500],
                "attempt": attempt,
            }
        except BaseException as error:
            duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
            error_summary = f"{error.__class__.__name__}: {error}"
            status = "stopped" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            self.store.update_task_status(
                task_id,
                status=status,
                error_summary=error_summary[:500],
                duration_seconds=duration_seconds,
            )
            self.logger.error(
                "Interrupted %s in %.2fs: %s\n%s",
                training_task.label,
                duration_seconds,
                error_summary,
                traceback.format_exc(),
            )
            clear_ml_memory()
            traceback_summary = self._traceback_summary(error)
            self._log_structured_phase(
                phase="after_cleanup",
                task_record=record,
                attempt=attempt,
                error=error,
                traceback_summary=traceback_summary,
            )
            self._log_structured_phase(
                phase="task:failed",
                task_record=record,
                attempt=attempt,
                message=f"Interrupted in {duration_seconds:.2f}s",
                error=error,
                traceback_summary=traceback_summary,
                extra={
                    "duration_seconds": duration_seconds,
                    "task_label": training_task.label,
                    "task_metadata": {
                        "id": record["id"],
                        "model_name": record["model_name"],
                        "preproc_id": record["preproc_id"],
                        "param_id": record["param_id"],
                        "param_display": record["param_display"],
                    },
                },
            )
            raise

    def _resolve_test_task_shim_action(
        self,
        *,
        record: dict[str, Any],
        training_task: TrainingTask,
    ) -> dict[str, Any] | None:
        shim_path_raw = os.environ.get(TEST_TASK_SHIM_PATH_ENV)
        if not shim_path_raw:
            return None

        payload = _load_test_task_shim(Path(shim_path_raw).expanduser())
        by_task_id = payload.get("by_task_id")
        if isinstance(by_task_id, dict):
            resolved = by_task_id.get(str(record["id"]))
            if isinstance(resolved, dict):
                return dict(resolved)

        by_preproc_id = payload.get("by_preproc_id")
        if isinstance(by_preproc_id, dict):
            resolved = by_preproc_id.get(str(training_task.preproc_id))
            if isinstance(resolved, dict):
                return dict(resolved)

        by_param_id = payload.get("by_param_id")
        if isinstance(by_param_id, dict):
            resolved = by_param_id.get(str(training_task.param_id))
            if isinstance(resolved, dict):
                return dict(resolved)

        default_action = payload.get("default")
        if isinstance(default_action, dict):
            return dict(default_action)
        return None

    def _execute_task_entry_with_test_shim(
        self,
        *,
        record: dict[str, Any],
        training_task: TrainingTask,
    ) -> dict[str, Any]:
        task_id = record["id"]
        action = self._resolve_test_task_shim_action(
            record=record,
            training_task=training_task,
        )
        if action is None:
            raise RuntimeError("Missing test task action while test shim is enabled.")

        self.logger.info("Running %s (test shim)", training_task.label)
        print(f"Running {training_task.label} (test shim)")
        updated_state = self.store.update_task_status(task_id, status="running")
        task_snapshot = next(task for task in updated_state["tasks"] if task["id"] == task_id)
        attempt = int(task_snapshot.get("attempts", 1))
        started_at = datetime.now(timezone.utc)

        self._log_structured_phase(
            phase="task:start",
            task_record=record,
            attempt=attempt,
            message=f"Running {training_task.label} (test shim)",
        )

        sleep_seconds = max(0.0, float(action.get("sleep_seconds", 0.0)))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
        status = str(action.get("status", "completed"))
        if status not in {"completed", "failed", "stopped"}:
            raise ValueError(f"Unsupported test shim status: {status}")

        if status == "completed":
            result_summary = action.get("result_summary")
            if not isinstance(result_summary, dict):
                result_summary = {"best_val_acc": float(action.get("best_val_acc", 0.0))}
            self.store.update_task_status(
                task_id,
                status="completed",
                result_summary=result_summary,
                duration_seconds=duration_seconds,
            )
            self._log_structured_phase(
                phase="after_cleanup",
                task_record=record,
                attempt=attempt,
            )
            self._log_structured_phase(
                phase="task:completed",
                task_record=record,
                attempt=attempt,
                message=f"Completed in {duration_seconds:.2f}s (test shim)",
                extra={"duration_seconds": duration_seconds},
            )
            return {
                "task_id": task_id,
                "status": "completed",
                "task_label": training_task.label,
                "duration_seconds": duration_seconds,
                "attempt": attempt,
            }

        failure_kind = action.get("failure_kind")
        error_summary = str(action.get("error_summary") or f"Synthetic {status} from test shim.")
        self.store.update_task_status(
            task_id,
            status=status,
            error_summary=error_summary[:500],
            duration_seconds=duration_seconds,
        )
        if failure_kind:
            self.store.update_task_execution_details(
                task_id,
                failure_kind=str(failure_kind),
            )
        self._log_structured_phase(
            phase="after_cleanup",
            task_record=record,
            attempt=attempt,
        )
        self._log_structured_phase(
            phase="task:failed" if status == "failed" else "task:stopped",
            task_record=record,
            attempt=attempt,
            message=f"{status.capitalize()} in {duration_seconds:.2f}s (test shim)",
            extra={
                "duration_seconds": duration_seconds,
                "task_label": training_task.label,
                "failure_kind": failure_kind,
            },
        )
        return {
            "task_id": task_id,
            "status": status,
            "task_label": training_task.label,
            "duration_seconds": duration_seconds,
            "error_summary": error_summary[:500],
            "attempt": attempt,
            "failure_kind": (None if not failure_kind else str(failure_kind)),
        }

    def run_one_task(
        self,
        task_id: str,
        *,
        task_cooldown_seconds: float = 0.0,
        max_queue_tasks: int | None = MAX_QUEUE_TASKS,
        queue_export_path: str | Path | None = None,
        task_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._task_cooldown_seconds = max(0.0, float(task_cooldown_seconds))
        if max_queue_tasks is not None and max_queue_tasks <= 0:
            raise ValueError("max_queue_tasks must be > 0 when provided.")
        self.store.ensure_dirs()
        isolated_child_mode = self._is_isolated_child_mode()
        if self.store.has_active_run():
            pid_record = self.store.read_pid_record() or {}
            pid = pid_record.get("pid")
            if self._pid_matches_current_process(pid):
                pid = None
            parent_pid = os.environ.get(ISOLATED_TASK_PARENT_PID_ENV)
            parent_matches = parent_pid is not None and str(pid) == parent_pid
            if pid is not None and not (isolated_child_mode and parent_matches):
                raise RuntimeError(f"Another experiment runner is already active with pid={pid}.")

        if task_record is None:
            queue_entries, state = self._build_queue_and_sync_state(
                max_queue_tasks=max_queue_tasks,
                queue_export_path=queue_export_path,
            )
            selected = self._find_task_in_queue(queue_entries, task_id)
            if selected is None:
                raise ValueError(f"Task id '{task_id}' was not found in the experiment queue for config '{self.config_path}'.")
            record, training_task = selected
        else:
            record = _apply_task_defaults(dict(task_record))
            if str(record.get("id")) != task_id:
                raise ValueError(f"Task id mismatch: expected '{task_id}', received '{record.get('id')}'.")
            training_task = build_training_task_from_record(record, self.training_config)
            state = self.store.register_task_record(record, config_path=self.config_path)
        state_task = next(task for task in state["tasks"] if task["id"] == task_id)
        resolved_status = str(state_task.get("status", "pending"))

        if resolved_status == "completed":
            artifacts_exist = task_has_existing_artifacts(self.training_config, training_task)
            reason = "already_completed"
            if self.training_config.run_skip and artifacts_exist:
                reason = "reconciled_from_existing_artifacts"
            message = f"Skipping {training_task.label} because it is already marked completed " f"(reason={reason})."
            self.logger.info(message)
            print(message)
            self._log_structured_phase(
                phase="task:skipped",
                task_record=record,
                message=message,
                extra={
                    "skip_reason": reason,
                    "artifacts_exist": artifacts_exist,
                    "run_skip_enabled": self.training_config.run_skip,
                },
            )
            return {
                "task_id": task_id,
                "status": "completed",
                "task_label": training_task.label,
                "skipped": True,
                "skip_reason": reason,
            }

        command = [sys.executable, "run_experiments.py", "run-task", "--task-id", task_id]
        if not isolated_child_mode:
            self.store.clear_stop_request()
            self.store.write_pid_record(config_path=self.config_path, command=command)
            self._install_signal_handlers()
        self._log_structured_phase(
            phase="run-task:start",
            task_record=record,
            message=f"Starting single-task run for {task_id}.",
            extra={"task_id": task_id},
        )
        self._apply_cpu_limits_for_current_process(record=record)

        result: dict[str, Any]
        try:
            if os.environ.get(TEST_TASK_SHIM_PATH_ENV):
                result = self._execute_task_entry_with_test_shim(
                    record=record,
                    training_task=training_task,
                )
            else:
                train_df = load_split_dataframe(self.training_config.train_split_path)
                test_df = load_split_dataframe(self.training_config.test_split_path)
                result = self._execute_task_entry(
                    record=record,
                    training_task=training_task,
                    train_df=train_df,
                    test_df=test_df,
                )
        finally:
            self._log_structured_phase(
                phase="run-task:finish",
                task_record=record,
                message=f"Single-task run finished for {task_id}.",
                extra={"task_id": task_id},
            )
            if not isolated_child_mode:
                self.store.clear_pid_record()
                self._restore_signal_handlers()

        snapshot = self.store.summarize()
        return {
            **result,
            "snapshot": snapshot,
            "peak_process_memory_mb": self._peak_process_memory_mb,
        }

    def run(self, options: IterativeRunOptions | None = None) -> dict[str, Any]:
        self.logger = self._build_logger()
        resolved_options = options or IterativeRunOptions()
        has_process_local_components = self.model_builders is not None or self.preprocessing_tasks is not None
        if resolved_options.isolate_tasks and has_process_local_components:
            self.logger.warning("isolate_tasks=True with process-local model_builders/preprocessing_tasks; " "ensure child process can resolve task dependencies.")
        elif not resolved_options.isolate_tasks and not has_process_local_components:
            self.logger.warning("Forcing isolate_tasks=True because device-policy handling and GPU/CPU recovery " "are only reliable with per-task subprocess isolation.")
            resolved_options = replace(resolved_options, isolate_tasks=True)
        self._task_cooldown_seconds = max(0.0, float(resolved_options.task_cooldown_seconds))
        if resolved_options.task_timeout_seconds is not None and resolved_options.task_timeout_seconds <= 0:
            raise ValueError("task_timeout_seconds must be > 0 when provided.")
        if resolved_options.device_policy not in DEVICE_POLICY_VALUES:
            raise ValueError("device_policy must be one of: gpu-first, cpu-only, gpu-only, adaptive.")
        if resolved_options.gpu_retries < 0:
            raise ValueError("gpu_retries must be >= 0.")
        if resolved_options.cpu_retries < 0:
            raise ValueError("cpu_retries must be >= 0.")
        if resolved_options.cooldown_after_oom_seconds < 0:
            raise ValueError("cooldown_after_oom_seconds must be >= 0.")
        if resolved_options.gpu_recovery_cooldown_seconds < 0:
            raise ValueError("gpu_recovery_cooldown_seconds must be >= 0.")
        if resolved_options.max_consecutive_oom <= 0:
            raise ValueError("max_consecutive_oom must be > 0.")
        if resolved_options.max_task_attempts <= 0:
            raise ValueError("max_task_attempts must be > 0.")
        if resolved_options.thermal_cooldown_seconds < 0:
            raise ValueError("thermal_cooldown_seconds must be >= 0.")
        if resolved_options.thermal_cpu_temp_celsius_limit is not None and resolved_options.thermal_cpu_temp_celsius_limit <= 0:
            raise ValueError("thermal_cpu_temp_celsius_limit must be > 0 when provided.")
        if resolved_options.thermal_gpu_temp_celsius_limit is not None and resolved_options.thermal_gpu_temp_celsius_limit <= 0:
            raise ValueError("thermal_gpu_temp_celsius_limit must be > 0 when provided.")
        if resolved_options.thermal_gpu_recovery_temp_celsius is not None and resolved_options.thermal_gpu_recovery_temp_celsius <= 0:
            raise ValueError("thermal_gpu_recovery_temp_celsius must be > 0 when provided.")
        if resolved_options.thermal_cpu_load_percent_limit is not None and not 0 <= resolved_options.thermal_cpu_load_percent_limit <= 100:
            raise ValueError("thermal_cpu_load_percent_limit must be between 0 and 100 when provided.")
        if resolved_options.thermal_gpu_utilization_percent_limit is not None and not 0 <= resolved_options.thermal_gpu_utilization_percent_limit <= 100:
            raise ValueError("thermal_gpu_utilization_percent_limit must be between 0 and 100 when provided.")
        if resolved_options.thermal_gpu_temp_celsius_limit is not None and resolved_options.thermal_gpu_recovery_temp_celsius is not None and resolved_options.thermal_gpu_recovery_temp_celsius > resolved_options.thermal_gpu_temp_celsius_limit:
            raise ValueError("thermal_gpu_recovery_temp_celsius must be <= thermal_gpu_temp_celsius_limit.")
        if resolved_options.max_queue_tasks is not None and resolved_options.max_queue_tasks <= 0:
            raise ValueError("max_queue_tasks must be > 0 when provided.")
        self.store.ensure_dirs()
        recovered = self.store.reconcile_for_launch()
        if recovered:
            self.logger.info(str(recovered["reason"]))
        if self.store.has_active_run():
            pid_record = self.store.read_pid_record() or {}
            pid = pid_record.get("pid")
            if not self._pid_matches_current_process(pid):
                raise RuntimeError(f"Another experiment runner is already active with pid={pid}.")

        estimated_counts = self.estimate_grid_counts()
        use_stream_queue_mode = resolved_options.max_queue_tasks is None and estimated_counts.total_experiments > STREAMING_QUEUE_TASK_THRESHOLD and not has_process_local_components
        if use_stream_queue_mode and (resolved_options.rerun_failed or resolved_options.rerun_completed):
            raise ValueError("rerun_failed/rerun_completed are not supported with streamed huge-queue execution. " "Use a targeted rerun instead.")
        self.store.set_expected_queue_totals(
            total_experiments=estimated_counts.total_experiments,
        )
        command = [sys.executable, "run_experiments.py", "launch", "--_launch-worker"]

        queue_entries: list[tuple[dict[str, Any], TrainingTask]] = []
        runnable_ids: list[str] = []
        runnable_ids_before_limit: list[str] = []
        if use_stream_queue_mode:
            state = self.store.load_state(config_path=self.config_path)
        else:
            queue_entries, state = self._build_queue_and_sync_state(
                max_queue_tasks=resolved_options.max_queue_tasks,
                queue_export_path=resolved_options.queue_export_path,
            )
            runnable_ids_before_limit = self.store.select_runnable_task_ids(
                state,
                rerun_failed=resolved_options.rerun_failed,
                rerun_completed=resolved_options.rerun_completed,
            )
            runnable_ids = list(runnable_ids_before_limit)
            if resolved_options.limit is not None:
                runnable_ids = runnable_ids[: resolved_options.limit]
        launch_context = self._build_launch_context(
            options=resolved_options,
            use_stream_queue_mode=use_stream_queue_mode,
            estimated_total_experiments=int(estimated_counts.total_experiments),
            state=state,
            runnable_ids_before_limit=runnable_ids_before_limit,
            runnable_ids_after_limit=runnable_ids,
        )
        runtime_state = dict(_default_runner_runtime())
        runtime_state.update(dict(state.get("runtime", {})))
        runtime_state["device_policy"] = resolved_options.device_policy
        runtime_state.setdefault("preferred_device", "gpu")
        runtime_state["gpu_visible_devices"] = self._resolve_gpu_visible_devices_for_policy(
            device_policy=resolved_options.device_policy,
        )
        runtime_state["isolate_tasks"] = bool(resolved_options.isolate_tasks)
        runtime_state["allow_huge_queue"] = resolved_options.max_queue_tasks is None
        runtime_state["max_queue_tasks"] = resolved_options.max_queue_tasks
        runtime_state["stream_queue_mode"] = use_stream_queue_mode
        runtime_state["launch_context"] = launch_context
        runtime_state["thermal_policy_enabled"] = bool(resolved_options.thermal_policy_enabled)
        runtime_state["thermal_state"] = "idle" if resolved_options.thermal_policy_enabled else "disabled"
        if resolved_options.device_policy == "cpu-only":
            runtime_state["preferred_device"] = "cpu"
            runtime_state["gpu_health"] = "unhealthy"
        elif resolved_options.device_policy == "gpu-only":
            runtime_state["preferred_device"] = "gpu"
            runtime_state["gpu_health"] = "healthy"
            self._validate_gpu_only_policy(runtime_state)
        self.store.update_runtime(runtime_state)

        if not use_stream_queue_mode and not runnable_ids:
            self.store.mark_launch_requested()
            self.logger.info("No pending experiments to run.")
            snapshot = self.store.summarize()
            snapshot["peak_process_memory_mb"] = self._peak_process_memory_mb
            return snapshot

        self.store.mark_launch_requested()
        self.store.write_pid_record(
            config_path=self.config_path,
            command=command,
            stdout_log_path=os.environ.get(BACKGROUND_STDOUT_LOG_ENV),
            stderr_log_path=os.environ.get(BACKGROUND_STDERR_LOG_ENV),
        )
        self._install_signal_handlers()
        planned_runnable_count = int(launch_context.get("planned_runnable_count", 0))
        self.logger.info(
            "Starting iterative run with %s runnable experiments (stream_queue_mode=%s).",
            planned_runnable_count,
            use_stream_queue_mode,
        )
        self._log_structured_phase(
            phase="run:start",
            message=(f"Starting run with {planned_runnable_count} runnable tasks " f"(stream_queue_mode={use_stream_queue_mode})."),
            extra={
                "runnable_count": planned_runnable_count,
                "stream_queue_mode": use_stream_queue_mode,
                "isolate_tasks": resolved_options.isolate_tasks,
                "task_cooldown_seconds": self._task_cooldown_seconds,
                "task_timeout_seconds": resolved_options.task_timeout_seconds,
                "device_policy": resolved_options.device_policy,
                "gpu_retries": resolved_options.gpu_retries,
                "cpu_retries": resolved_options.cpu_retries,
                "cooldown_after_oom_seconds": resolved_options.cooldown_after_oom_seconds,
                "gpu_recovery_cooldown_seconds": resolved_options.gpu_recovery_cooldown_seconds,
                "max_consecutive_oom": resolved_options.max_consecutive_oom,
                "max_task_attempts": resolved_options.max_task_attempts,
                "fail_fast_on_oom": resolved_options.fail_fast_on_oom,
                "thermal_policy_enabled": resolved_options.thermal_policy_enabled,
                "thermal_cpu_temp_celsius_limit": resolved_options.thermal_cpu_temp_celsius_limit,
                "thermal_cpu_load_percent_limit": resolved_options.thermal_cpu_load_percent_limit,
                "thermal_gpu_temp_celsius_limit": resolved_options.thermal_gpu_temp_celsius_limit,
                "thermal_gpu_utilization_percent_limit": resolved_options.thermal_gpu_utilization_percent_limit,
                "thermal_gpu_recovery_temp_celsius": resolved_options.thermal_gpu_recovery_temp_celsius,
                "thermal_cooldown_seconds": resolved_options.thermal_cooldown_seconds,
                "max_queue_tasks": resolved_options.max_queue_tasks,
                "queue_export_path": resolved_options.queue_export_path,
                "launch_id": launch_context.get("launch_id"),
                "launch_skipped_count": launch_context.get("skipped_count"),
                "launch_skip_reason_counts": launch_context.get("skip_reason_counts"),
            },
        )

        try:
            runnable_set = set(runnable_ids)
            if resolved_options.isolate_tasks:
                queue_iterable = self._iter_queue_entries() if use_stream_queue_mode else queue_entries
                streamed_started = 0
                for record, _training_task in queue_iterable:
                    task_id = record["id"]
                    if not use_stream_queue_mode and task_id not in runnable_set:
                        continue
                    if use_stream_queue_mode and resolved_options.limit is not None and streamed_started >= resolved_options.limit:
                        break
                    if self.store.stop_requested():
                        self.logger.info(
                            "Stop requested before starting %s. Ending current run.",
                            task_id,
                        )
                        break

                    task_result = self._run_task_with_isolated_device_policy(
                        record=record,
                        options=resolved_options,
                        runtime_state=runtime_state,
                    )
                    if use_stream_queue_mode:
                        streamed_started += 1
                    self.store.update_runtime(runtime_state)
                    if str(task_result.get("status", "failed")) != "completed":
                        final_failure_kind = str(task_result.get("failure_kind") or "")
                        is_final_oom = final_failure_kind in {"oom", "gpu_oom", "cpu_oom"}
                        if is_final_oom and int(runtime_state.get("consecutive_final_oom_failures", 0)) >= int(resolved_options.max_consecutive_oom):
                            runtime_state["oom_policy_stop"] = "Reached max_consecutive_oom=" f"{resolved_options.max_consecutive_oom} after task {task_id}."
                            self.store.update_runtime(runtime_state)
                            self._log_structured_phase(
                                phase="run:oom_policy_stop",
                                event="oom_policy_stop",
                                task_record=record,
                                message=str(runtime_state["oom_policy_stop"]),
                                extra={
                                    "task_id": task_id,
                                    "consecutive_final_oom_failures": runtime_state.get("consecutive_final_oom_failures"),
                                    "max_consecutive_oom": resolved_options.max_consecutive_oom,
                                },
                            )
                            self.store.request_stop(reason="oom_policy_stop")
                        elif not is_final_oom:
                            runtime_state["consecutive_final_oom_failures"] = 0
                            self.store.update_runtime(runtime_state)
                    else:
                        runtime_state["consecutive_final_oom_failures"] = 0
                        self.store.update_runtime(runtime_state)

                    if self.store.stop_requested():
                        self.logger.info(
                            "Stop requested after finishing %s. Ending current run.",
                            task_id,
                        )
                        break
            else:
                train_df = load_split_dataframe(self.training_config.train_split_path)
                test_df = load_split_dataframe(self.training_config.test_split_path)
                queue_iterable = self._iter_queue_entries() if use_stream_queue_mode else queue_entries
                streamed_started = 0
                for record, training_task in queue_iterable:
                    task_id = record["id"]
                    if not use_stream_queue_mode and task_id not in runnable_set:
                        continue
                    if use_stream_queue_mode and resolved_options.limit is not None and streamed_started >= resolved_options.limit:
                        break
                    if self.store.stop_requested():
                        self.logger.info(
                            "Stop requested before starting %s. Ending current run.",
                            task_id,
                        )
                        break

                    if use_stream_queue_mode:
                        self.store.register_task_record(record, config_path=self.config_path)
                    self._execute_task_entry(
                        record=record,
                        training_task=training_task,
                        train_df=train_df,
                        test_df=test_df,
                    )
                    if use_stream_queue_mode:
                        streamed_started += 1

                    if self.store.stop_requested():
                        self.logger.info(
                            "Stop requested after finishing %s. Ending current run.",
                            task_id,
                        )
                        break
        finally:
            self._log_structured_phase(phase="run:finish", message="Run finished.")
            self.store.clear_pid_record()
            self._restore_signal_handlers()
            self._close_logger()

        snapshot = self.store.summarize()
        snapshot["peak_process_memory_mb"] = self._peak_process_memory_mb
        return snapshot


def run_one_experiment_task(
    *,
    task_id: str,
    config_path: Path,
    project_paths: ProjectPaths,
    training_config: TrainingConfig,
    task_cooldown_seconds: float = 0.0,
    max_queue_tasks: int | None = MAX_QUEUE_TASKS,
    queue_export_path: str | Path | None = None,
    task_record: dict[str, Any] | None = None,
    cpu_execution_limits: CpuExecutionLimits | None = None,
    model_builders: dict[str, ModelBuilder] | None = None,
    preprocessing_tasks: Iterable[Any] | None = None,
) -> dict[str, Any]:
    runner = IterativeExperimentRunner(
        config_path=config_path,
        project_paths=project_paths,
        training_config=training_config,
        cpu_execution_limits=cpu_execution_limits,
        model_builders=model_builders,
        preprocessing_tasks=preprocessing_tasks,
    )
    return runner.run_one_task(
        task_id,
        task_cooldown_seconds=task_cooldown_seconds,
        max_queue_tasks=max_queue_tasks,
        queue_export_path=queue_export_path,
        task_record=task_record,
    )


def launch_background_runner(
    *,
    script_path: Path,
    forwarded_args: list[str],
    cwd: Path,
    logs_dir: Path,
) -> BackgroundRunnerLaunch:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = _build_background_log_paths(logs_dir)
    command = [sys.executable, str(script_path), "launch", "--_launch-worker", *forwarded_args]
    stdout_handle = stdout_path.open("a", encoding="utf-8")
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env[BACKGROUND_STDOUT_LOG_ENV] = str(stdout_path)
    env[BACKGROUND_STDERR_LOG_ENV] = str(stderr_path)
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "env": env,
        "stdout": stdout_handle,
        "stderr": stderr_handle,
    }
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if creationflags:
            popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **popen_kwargs)
    finally:
        stdout_handle.close()
        stderr_handle.close()

    return BackgroundRunnerLaunch(
        process=process,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
