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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pipeline.experiments.failures import classify_task_failure
from pipeline.train.models import MODEL_BUILDERS, ModelBuilder
from pipeline.train.preprocessing import count_preprocessing_tasks
from pipeline.train.runner import (
    TrainingConfig,
    TrainingTask,
    artifact_paths_for_task,
    build_history_row,
    build_training_tasks,
    load_split_dataframe,
    run_training_task,
    save_run_result,
    task_has_existing_artifacts,
)
from pipeline.utils.memory import clear_ml_memory, log_memory_snapshot
from pipeline.utils.paths import ProjectPaths

STATE_SCHEMA_VERSION = 2
RUNNER_STATE_FILENAME = "runner_state.json"
RUNNER_SUMMARY_FILENAME = "experiment_runs.csv"
RUNNER_LOG_FILENAME = "iterative-runner.log"
RUN_EVENTS_FILENAME = "run-events.jsonl"
TASK_LOGS_DIRNAME = "tasks"
RUNNER_PID_FILENAME = "runner_pid.json"
STOP_REQUEST_FILENAME = "stop_requested.flag"
BACKGROUND_RUNNER_LOG_PREFIX = "background-runner"
TASK_STATUSES = {"pending", "running", "completed", "failed", "stopped"}
MAX_QUEUE_TASKS = 50_000
ISOLATED_TASK_CHILD_MODE_ENV = "BREAST_CANCER_ANALYSIS_ISOLATED_TASK_CHILD"
ISOLATED_TASK_PARENT_PID_ENV = "BREAST_CANCER_ANALYSIS_ISOLATED_TASK_PARENT_PID"
DEVICE_POLICY_VALUES = {"gpu-first", "cpu-only", "gpu-only", "adaptive"}
GPU_HEALTH_STATES = {"healthy", "cooling_down", "unhealthy"}


@dataclass(frozen=True)
class IterativeRunOptions:
    rerun_failed: bool = False
    rerun_completed: bool = False
    limit: int | None = None
    isolate_tasks: bool = False
    task_cooldown_seconds: float = 2.0
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


@dataclass(frozen=True)
class BackgroundRunnerLaunch:
    process: subprocess.Popen[Any]
    stdout_path: Path
    stderr_path: Path


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
        "oom_policy_stop": None,
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
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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
        handle.write(json.dumps(_normalize_json_value(payload), sort_keys=True) + os.linesep)


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        "parameters": {
            "model_name": task.model_name,
            "preprocessing_id": task.preproc_id,
            "preprocessing_params": json.loads(task.param_json),
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

    def has_active_run(self) -> bool:
        pid_record = self.read_pid_record()
        pid = None if pid_record is None else pid_record.get("pid")
        return _is_process_alive(pid)

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

    def clear_pid_record(self) -> None:
        if self.pid_path.exists():
            self.pid_path.unlink()

    def request_stop(self, *, reason: str = "manual") -> Path:
        _atomic_write_text(
            self.stop_flag_path,
            json.dumps({"requested_at": _timestamp_now(), "reason": reason}),
        )
        return self.stop_flag_path

    def clear_stop_request(self) -> None:
        if self.stop_flag_path.exists():
            self.stop_flag_path.unlink()

    def stop_requested(self) -> bool:
        return self.stop_flag_path.exists()

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
        self._persist_state(state)
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

    def _persist_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _timestamp_now()
        _atomic_write_json(self.state_path, state)
        self._write_summary_csv(state)
        self._write_task_snapshots(state)

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
        now = _timestamp_now()
        task_found = False
        for task in state["tasks"]:
            if task["id"] != task_id:
                continue

            task_found = True
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
            break

        if not task_found:
            raise KeyError(f"Task id not found in runner state: {task_id}")

        self._persist_state(state)
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
        self._persist_state(state)
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
        self._persist_state(state)
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
        self._persist_state(state)
        return state

    def summarize(self) -> dict[str, Any]:
        state = self.load_state()
        pid_record = self.read_pid_record()
        active_pid = None if pid_record is None else pid_record.get("pid")
        stdout_log_path = None if pid_record is None else pid_record.get("stdout_log_path")
        stderr_log_path = None if pid_record is None else pid_record.get("stderr_log_path")
        active_run = _is_process_alive(active_pid)
        if not active_run and any(task.get("status") == "running" for task in state["tasks"]):
            self._reconcile_running_tasks(state)
            self._persist_state(state)

        counts = {status: 0 for status in TASK_STATUSES}
        for task in state["tasks"]:
            status = task.get("status", "pending")
            counts[status] = counts.get(status, 0) + 1

        current_task = None
        for task in state["tasks"]:
            if task["id"] == state.get("current_task_id"):
                current_task = task
                break

        total = len(state["tasks"])
        stop_requested = self.stop_requested()
        if active_run and stop_requested:
            overall_status = "stopping"
        elif active_run:
            overall_status = "running"
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
        runtime = dict(_default_runner_runtime())
        runtime.update(dict(state.get("runtime", {})))

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
            "device_policy": runtime.get("device_policy"),
            "preferred_device": runtime.get("preferred_device"),
            "gpu_health": runtime.get("gpu_health"),
            "last_gpu_oom_task_id": runtime.get("last_gpu_oom_task_id"),
            "gpu_oom_count": runtime.get("gpu_oom_count"),
            "cpu_fallback_successes": runtime.get("cpu_fallback_successes"),
            "consecutive_final_oom_failures": runtime.get("consecutive_final_oom_failures"),
            "last_successful_device": runtime.get("last_successful_device"),
            "oom_policy_stop": runtime.get("oom_policy_stop"),
        }

    def reset(self, *, purge_results: bool = False) -> None:
        self.clear_stop_request()
        self.clear_pid_record()

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


class IterativeExperimentRunner:
    def __init__(
        self,
        *,
        config_path: Path,
        project_paths: ProjectPaths,
        training_config: TrainingConfig,
        model_builders: dict[str, ModelBuilder] | None = None,
        preprocessing_tasks: Iterable[Any] | None = None,
    ) -> None:
        self.config_path = config_path
        self.project_paths = project_paths
        self.training_config = training_config
        self.model_builders = model_builders
        self.preprocessing_tasks = preprocessing_tasks
        self.store = ExperimentStateStore(project_paths)
        self.logger = self._build_logger()
        self._previous_signal_handlers: dict[int, Any] = {}
        self._peak_process_memory_mb: float | None = None
        self._task_cooldown_seconds: float = 2.0

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"iterative_experiment_runner:{self.project_paths.artifacts_dir}")
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        logger.propagate = False
        self.store.log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.store.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger

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
            handle.write(line + os.linesep)

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
        available_builders = self.model_builders or MODEL_BUILDERS
        model_names = self.training_config.model_names or list(available_builders.keys())
        resolved_preprocessing_tasks = list(self.preprocessing_tasks) if self.preprocessing_tasks is not None else None
        preprocessing_count = (
            len(resolved_preprocessing_tasks)
            if resolved_preprocessing_tasks is not None
            else count_preprocessing_tasks(
                self.training_config.preprocessing_ids,
                include_combinations=self.training_config.include_combinations,
                param_grids=self.training_config.preprocessing_grids,
            )
        )
        estimated_task_count = preprocessing_count * len(model_names)
        if estimated_task_count > MAX_QUEUE_TASKS:
            raise ValueError("The requested experiment grid expands to " f"{estimated_task_count:,} training tasks, which exceeds the " f"safety limit of {MAX_QUEUE_TASKS:,}. Narrow the run with " "`--models`, `--preprocessing`, `--no-combined-preprocessing`, " "or a smaller preprocessing grid.")

        training_tasks = build_training_tasks(
            self.training_config,
            model_builders=self.model_builders,
            preprocessing_tasks=resolved_preprocessing_tasks,
        )
        return [(build_experiment_record(task, self.training_config), task) for task in training_tasks]

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

    def _build_queue_and_sync_state(self) -> tuple[list[tuple[dict[str, Any], TrainingTask]], dict[str, Any]]:
        queue_entries = self.build_queue()
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

    def _task_state_by_id(self, task_id: str) -> dict[str, Any]:
        state = self.store.load_state()
        for task in state["tasks"]:
            if task.get("id") == task_id:
                return task
        raise KeyError(f"Task id not found in runner state: {task_id}")

    def _run_task_subprocess_command(self, task_id: str, command_base: tuple[str, ...] | None) -> list[str]:
        if command_base is None:
            command = [sys.executable, "run_experiments.py", "run-task"]
        else:
            command = [str(entry) for entry in command_base]
        return [*command, "--task-id", task_id]

    def _execute_task_entry_isolated_subprocess(
        self,
        *,
        record: dict[str, Any],
        command_base: tuple[str, ...] | None,
        timeout_seconds: float | None,
        attempt_number: int,
        device: str,
        gpu_visible_devices: str | None,
    ) -> dict[str, Any]:
        task_id = record["id"]
        command = self._run_task_subprocess_command(task_id, command_base)
        env = os.environ.copy()
        env[ISOLATED_TASK_CHILD_MODE_ENV] = "1"
        env[ISOLATED_TASK_PARENT_PID_ENV] = str(os.getpid())
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            if gpu_visible_devices is None:
                env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = gpu_visible_devices
        started_at = datetime.now(timezone.utc)
        self.logger.info(
            "Running isolated task %s via subprocess (attempt=%s, device=%s): %s",
            task_id,
            attempt_number,
            device,
            " ".join(command),
        )
        print(f"Running isolated task {task_id} (attempt={attempt_number}, device={device})")
        self._log_structured_phase(
            phase="task:subprocess_start",
            task_record=record,
            message=f"Launching isolated subprocess for {task_id}.",
            attempt=attempt_number,
            extra={
                "subprocess_command": command,
                "task_timeout_seconds": timeout_seconds,
                "device": device,
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
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
                device=device,
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
                    "device": device,
                    "exit_code": None,
                    "failure_kind": failure_kind,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at,
                    "duration_seconds": duration_seconds,
                },
            )
            self.store.update_task_execution_details(
                task_id,
                final_device=device,
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
                    "device": device,
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
                "device": device,
                "attempt": attempt_number,
            }

        duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
        exit_code = int(completed.returncode)
        task_state = self._task_state_by_id(task_id)
        task_status = str(task_state.get("status", "pending"))
        task_events_path, _task_memory_path, task_log_path = self._task_log_paths(task_id)
        error_summary = task_state.get("error_summary")
        failure_kind = classify_task_failure(
            device=device,
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
                "device": device,
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
                final_device=device,
                failure_kind=None,
            )
        else:
            self.store.update_task_execution_details(
                task_id,
                final_device=device,
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
                "device": device,
                "failure_kind": (None if task_status == "completed" else failure_kind),
            },
        )
        if exit_code == 0:
            print(f"Task {task_id} finished with status={task_status} (device={device})")
        else:
            print(f"Task {task_id} subprocess failed with exit code {exit_code} (device={device})")
        return {
            "task_id": task_id,
            "status": task_status,
            "duration_seconds": duration_seconds,
            "exit_code": exit_code,
            "timeout": False,
            "failure_kind": (None if task_status == "completed" else failure_kind),
            "device": device,
            "attempt": attempt_number,
        }

    def _is_isolated_child_mode(self) -> bool:
        return os.environ.get(ISOLATED_TASK_CHILD_MODE_ENV) == "1"

    def _sleep_with_log(self, seconds: float, *, reason: str) -> None:
        delay = max(0.0, float(seconds))
        if delay <= 0:
            return
        self.logger.info("Sleeping for %.2fs (%s).", delay, reason)
        time.sleep(delay)

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
        attempt_results: list[dict[str, Any]] = []
        attempted_devices: list[str] = []
        gpu_retries_remaining = int(options.gpu_retries)
        cpu_retries_remaining = int(options.cpu_retries)
        total_attempts = 0
        next_device = "cpu" if policy == "cpu-only" else str(runtime_state.get("preferred_device", "gpu"))

        if policy == "gpu-only":
            next_device = "gpu"
        if policy == "gpu-first":
            next_device = "gpu"
        if policy == "adaptive":
            cooldown_until = _parse_timestamp(runtime_state.get("gpu_recovery_cooldown_until"))
            if next_device == "cpu" and cooldown_until is not None and datetime.now(timezone.utc) >= cooldown_until:
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
            attempt_result = self._execute_task_entry_isolated_subprocess(
                record=record,
                command_base=options.run_task_command_base,
                timeout_seconds=options.task_timeout_seconds,
                attempt_number=total_attempts,
                device=next_device,
                gpu_visible_devices=runtime_state.get("gpu_visible_devices"),
            )
            attempt_results.append(attempt_result)
            attempted_devices.append(next_device)
            failure_kind = attempt_result.get("failure_kind")
            task_status = str(attempt_result.get("status", "failed"))
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
                    if "gpu" in attempted_devices:
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
                        "exit_code": attempt_result.get("exit_code"),
                        "failure_kind": None,
                        "fallback_next": None,
                        "cooldown_seconds": 0,
                    },
                )
                return {
                    **attempt_result,
                    "attempts": attempt_results,
                    "final_device": next_device,
                    "failure_kind": None,
                }

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
                        "exit_code": attempt_result.get("exit_code"),
                        "failure_kind": failure_kind,
                        "fallback_next": fallback_next,
                        "cooldown_seconds": cooldown_seconds,
                    },
                )
                self._sleep_with_log(cooldown_seconds, reason="gpu_oom_retry")
                next_device = "gpu"
                continue

            if next_device == "gpu" and is_any_oom and policy in {"gpu-first", "adaptive"}:
                if cpu_retries_remaining >= 0 and total_attempts < int(options.max_task_attempts):
                    fallback_next = "cpu"
                    runtime_state["gpu_health"] = "unhealthy"
                    cooldown_until = datetime.now(timezone.utc).timestamp() + float(options.gpu_recovery_cooldown_seconds)
                    runtime_state["gpu_recovery_cooldown_until"] = datetime.fromtimestamp(
                        cooldown_until,
                        tz=timezone.utc,
                    ).isoformat()
                    self.store.update_task_execution_details(
                        task_id,
                        fallback_reason="gpu_oom",
                    )
                    self._log_structured_phase(
                        phase="task:cpu_fallback_scheduled",
                        event="cpu_fallback_scheduled",
                        task_record=record,
                        attempt=total_attempts,
                        message=f"Scheduling CPU fallback for {task_id}.",
                    )
                    self._log_structured_phase(
                        phase="task:attempt_finished",
                        event="task_attempt_finished",
                        task_record=record,
                        attempt=total_attempts,
                        extra={
                            "task_id": task_id,
                            "device": next_device,
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
            "final_device": final_result.get("device"),
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

    def run_one_task(
        self,
        task_id: str,
        *,
        task_cooldown_seconds: float = 2.0,
    ) -> dict[str, Any]:
        self._task_cooldown_seconds = max(0.0, float(task_cooldown_seconds))
        self.store.ensure_dirs()
        isolated_child_mode = self._is_isolated_child_mode()
        if self.store.has_active_run():
            pid_record = self.store.read_pid_record() or {}
            pid = pid_record.get("pid")
            parent_pid = os.environ.get(ISOLATED_TASK_PARENT_PID_ENV)
            parent_matches = parent_pid is not None and str(pid) == parent_pid
            if not (isolated_child_mode and parent_matches):
                raise RuntimeError(f"Another experiment runner is already active with pid={pid}.")

        queue_entries, state = self._build_queue_and_sync_state()
        selected = self._find_task_in_queue(queue_entries, task_id)
        if selected is None:
            raise ValueError(f"Task id '{task_id}' was not found in the experiment queue for config '{self.config_path}'.")
        record, training_task = selected
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

        self.store.clear_stop_request()
        command = [sys.executable, "run_experiments.py", "run-task", "--task-id", task_id]
        if not isolated_child_mode:
            self.store.write_pid_record(config_path=self.config_path, command=command)
            self._install_signal_handlers()
        self._log_structured_phase(
            phase="run-task:start",
            task_record=record,
            message=f"Starting single-task run for {task_id}.",
            extra={"task_id": task_id},
        )

        result: dict[str, Any]
        try:
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
        resolved_options = options or IterativeRunOptions()
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
        self.store.ensure_dirs()
        if self.store.has_active_run():
            pid_record = self.store.read_pid_record() or {}
            pid = pid_record.get("pid")
            raise RuntimeError(f"Another experiment runner is already active with pid={pid}.")

        queue_entries, state = self._build_queue_and_sync_state()
        runnable_ids = self.store.select_runnable_task_ids(
            state,
            rerun_failed=resolved_options.rerun_failed,
            rerun_completed=resolved_options.rerun_completed,
        )
        if resolved_options.limit is not None:
            runnable_ids = runnable_ids[: resolved_options.limit]
        runtime_state = dict(_default_runner_runtime())
        runtime_state.update(dict(state.get("runtime", {})))
        runtime_state["device_policy"] = resolved_options.device_policy
        runtime_state.setdefault("preferred_device", "gpu")
        runtime_state["gpu_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        if resolved_options.device_policy == "cpu-only":
            runtime_state["preferred_device"] = "cpu"
            runtime_state["gpu_health"] = "unhealthy"
        elif resolved_options.device_policy == "gpu-only":
            runtime_state["preferred_device"] = "gpu"
            runtime_state["gpu_health"] = "healthy"
        self.store.update_runtime(runtime_state)

        if not runnable_ids:
            self.logger.info("No pending experiments to run.")
            snapshot = self.store.summarize()
            snapshot["peak_process_memory_mb"] = self._peak_process_memory_mb
            return snapshot

        self.store.clear_stop_request()
        command = [sys.executable, "run_experiments.py", "run"]
        self.store.write_pid_record(config_path=self.config_path, command=command)
        self._install_signal_handlers()
        self.logger.info(
            "Starting iterative run with %s runnable experiments.",
            len(runnable_ids),
        )
        self._log_structured_phase(
            phase="run:start",
            message=f"Starting run with {len(runnable_ids)} runnable tasks.",
            extra={
                "runnable_count": len(runnable_ids),
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
            },
        )

        try:
            runnable_set = set(runnable_ids)
            if resolved_options.isolate_tasks:
                for record, _training_task in queue_entries:
                    task_id = record["id"]
                    if task_id not in runnable_set:
                        continue
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
                for record, training_task in queue_entries:
                    task_id = record["id"]
                    if task_id not in runnable_set:
                        continue
                    if self.store.stop_requested():
                        self.logger.info(
                            "Stop requested before starting %s. Ending current run.",
                            task_id,
                        )
                        break

                    self._execute_task_entry(
                        record=record,
                        training_task=training_task,
                        train_df=train_df,
                        test_df=test_df,
                    )

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

        snapshot = self.store.summarize()
        snapshot["peak_process_memory_mb"] = self._peak_process_memory_mb
        return snapshot


def run_one_experiment_task(
    *,
    task_id: str,
    config_path: Path,
    project_paths: ProjectPaths,
    training_config: TrainingConfig,
    task_cooldown_seconds: float = 2.0,
    model_builders: dict[str, ModelBuilder] | None = None,
    preprocessing_tasks: Iterable[Any] | None = None,
) -> dict[str, Any]:
    runner = IterativeExperimentRunner(
        config_path=config_path,
        project_paths=project_paths,
        training_config=training_config,
        model_builders=model_builders,
        preprocessing_tasks=preprocessing_tasks,
    )
    return runner.run_one_task(
        task_id,
        task_cooldown_seconds=task_cooldown_seconds,
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
    command = [sys.executable, str(script_path), "run", *forwarded_args]
    stdout_handle = stdout_path.open("a", encoding="utf-8")
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            start_new_session=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()

    return BackgroundRunnerLaunch(
        process=process,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
