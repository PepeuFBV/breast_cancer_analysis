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
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pipeline.train.models import ModelBuilder
from pipeline.train.runner import (
    TrainingConfig,
    TrainingTask,
    artifact_paths_for_task,
    build_history_row,
    build_training_tasks,
    load_split_dataframe,
    run_training_task,
    save_run_result,
)
from pipeline.utils.paths import ProjectPaths

STATE_SCHEMA_VERSION = 1
RUNNER_STATE_FILENAME = "runner_state.json"
RUNNER_SUMMARY_FILENAME = "experiment_runs.csv"
RUNNER_LOG_FILENAME = "iterative-runner.log"
RUNNER_PID_FILENAME = "runner_pid.json"
STOP_REQUEST_FILENAME = "stop_requested.flag"
TASK_STATUSES = {"pending", "running", "completed", "failed", "stopped"}


@dataclass(frozen=True)
class IterativeRunOptions:
    rerun_failed: bool = False
    rerun_completed: bool = False
    limit: int | None = None


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_from_path(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


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
        return {
            str(key): _normalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
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


def _model_runtime_signature(
    config: TrainingConfig, model_name: str
) -> dict[str, Any] | None:
    runtime = (config.model_runtime or {}).get(model_name)
    if runtime is None:
        return None
    return _normalize_json_value(asdict(runtime))


def build_experiment_id(task: TrainingTask, config: TrainingConfig) -> str:
    payload = {
        "model_name": task.model_name,
        "preproc_id": task.preproc_id,
        "param_id": task.param_id,
        "param_json": json.loads(task.param_json),
        "train_split_path": str(config.train_split_path.resolve()),
        "test_split_path": str(config.test_split_path.resolve()),
        "folds": config.folds,
        "validation_size": config.validation_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "loss": config.loss,
        "random_state": config.random_state,
        "model_runtime": _model_runtime_signature(config, task.model_name),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"exp-{digest[:16]}"


def build_experiment_record(
    task: TrainingTask, config: TrainingConfig
) -> dict[str, Any]:
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
        "error_summary": None,
        "result_summary": {},
    }


class ExperimentStateStore:
    def __init__(self, project_paths: ProjectPaths) -> None:
        self.project_paths = project_paths
        self.state_path = project_paths.experiment_state_dir / RUNNER_STATE_FILENAME
        self.summary_path = (
            project_paths.experiment_summary_dir / RUNNER_SUMMARY_FILENAME
        )
        self.log_path = project_paths.experiment_logs_dir / RUNNER_LOG_FILENAME
        self.pid_path = project_paths.experiment_control_dir / RUNNER_PID_FILENAME
        self.stop_flag_path = (
            project_paths.experiment_control_dir / STOP_REQUEST_FILENAME
        )

    def ensure_dirs(self) -> None:
        self.project_paths.ensure_artifact_dirs()

    def load_state(self, *, config_path: Path | None = None) -> dict[str, Any]:
        state = _load_json_file(self.state_path) or {}
        state.setdefault("schema_version", STATE_SCHEMA_VERSION)
        state.setdefault("created_at", _timestamp_now())
        state["updated_at"] = state.get("updated_at", state["created_at"])
        state["config_path"] = (
            str(config_path) if config_path else state.get("config_path")
        )
        state["artifacts_dir"] = str(self.project_paths.artifacts_dir)
        state["history_dir"] = str(self.project_paths.history_dir)
        state["predictions_dir"] = str(self.project_paths.predictions_dir)
        state["summary_path"] = str(self.summary_path)
        state["current_task_id"] = state.get("current_task_id")
        state["tasks"] = list(state.get("tasks", []))
        return state

    def read_pid_record(self) -> dict[str, Any] | None:
        return _load_json_file(self.pid_path)

    def has_active_run(self) -> bool:
        pid_record = self.read_pid_record()
        pid = None if pid_record is None else pid_record.get("pid")
        return _is_process_alive(pid)

    def write_pid_record(self, *, config_path: Path, command: list[str]) -> None:
        _atomic_write_json(
            self.pid_path,
            {
                "pid": os.getpid(),
                "config_path": str(config_path),
                "command": command,
                "started_at": _timestamp_now(),
            },
        )

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
        existing_by_id = {
            task["id"]: task
            for task in state["tasks"]
            if isinstance(task, dict) and task.get("id")
        }
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
                    "error_summary": existing.get("error_summary"),
                    "result_summary": dict(existing.get("result_summary", {})),
                }
            )

        state["tasks"] = synced_tasks
        state["config_path"] = str(config_path)
        state["queue_signature"] = _queue_signature(synced_tasks)
        state["updated_at"] = now
        state["current_task_id"] = None

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
                task["error_summary"] = (
                    task.get("error_summary")
                    or "Runner interrupted before experiment completion."
                )

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
                task["error_summary"] = (
                    "Persisted state referenced missing artifacts; task re-queued."
                )
                task["result_summary"] = {}
                continue

            if not artifacts_exist:
                continue

            if task.get("status") in {"pending", "running", "stopped"}:
                task["status"] = "completed"
                task["updated_at"] = now
                task["finished_at"] = task.get("finished_at") or _timestamp_from_path(
                    history_path
                )
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
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
            "updated_at": task.get("updated_at"),
            "last_duration_seconds": task.get("last_duration_seconds"),
            "error_summary": task.get("error_summary"),
            "history_path": task.get("artifacts", {}).get("history_path"),
            "predictions_path": task.get("artifacts", {}).get("predictions_path"),
            "parameters_json": json.dumps(
                task.get("parameters", {}), sort_keys=True, separators=(",", ":")
            ),
        }
        row.update(
            {
                f"result_{key}": value
                for key, value in task.get("result_summary", {}).items()
            }
        )
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
        for task in state["tasks"]:
            if task["id"] != task_id:
                continue

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

        self._persist_state(state)
        return state

    def set_task_artifacts(
        self, task_id: str, *, history_path: Path, predictions_path: Path
    ) -> dict[str, Any]:
        state = self.load_state()
        for task in state["tasks"]:
            if task["id"] == task_id:
                task["artifacts"] = {
                    "history_path": str(history_path),
                    "predictions_path": str(predictions_path),
                }
                task["updated_at"] = _timestamp_now()
                break
        self._persist_state(state)
        return state

    def summarize(self) -> dict[str, Any]:
        state = self.load_state()
        pid_record = self.read_pid_record()
        active_pid = None if pid_record is None else pid_record.get("pid")
        active_run = _is_process_alive(active_pid)
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
            "config_path": state.get("config_path"),
            "updated_at": state.get("updated_at"),
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

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(
            f"iterative_experiment_runner:{self.project_paths.artifacts_dir}"
        )
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(self.store.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger

    def build_queue(self) -> list[tuple[dict[str, Any], TrainingTask]]:
        training_tasks = build_training_tasks(
            self.training_config,
            model_builders=self.model_builders,
            preprocessing_tasks=self.preprocessing_tasks,
        )
        return [
            (build_experiment_record(task, self.training_config), task)
            for task in training_tasks
        ]

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

    def run(self, options: IterativeRunOptions | None = None) -> dict[str, Any]:
        resolved_options = options or IterativeRunOptions()
        self.store.ensure_dirs()
        if self.store.has_active_run():
            pid_record = self.store.read_pid_record() or {}
            pid = pid_record.get("pid")
            raise RuntimeError(
                f"Another experiment runner is already active with pid={pid}."
            )

        queue_entries = self.build_queue()
        state = self.store.sync_queue(
            [record for record, _ in queue_entries], config_path=self.config_path
        )
        runnable_ids = self.store.select_runnable_task_ids(
            state,
            rerun_failed=resolved_options.rerun_failed,
            rerun_completed=resolved_options.rerun_completed,
        )
        if resolved_options.limit is not None:
            runnable_ids = runnable_ids[: resolved_options.limit]

        if not runnable_ids:
            self.logger.info("No pending experiments to run.")
            return self.store.summarize()

        self.store.clear_stop_request()
        command = [sys.executable, "run_experiments.py", "run"]
        self.store.write_pid_record(config_path=self.config_path, command=command)
        self._install_signal_handlers()
        self.logger.info(
            "Starting iterative run with %s runnable experiments.",
            len(runnable_ids),
        )

        try:
            train_df = load_split_dataframe(self.training_config.train_split_path)
            test_df = load_split_dataframe(self.training_config.test_split_path)
            runnable_set = set(runnable_ids)
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

                self.logger.info("Running %s", training_task.label)
                print(f"Running {training_task.label}")
                self.store.update_task_status(task_id, status="running")
                started_at = datetime.now(timezone.utc)

                try:
                    result = run_training_task(
                        training_task,
                        self.training_config,
                        train_df=train_df,
                        test_df=test_df,
                        model_builders=self.model_builders,
                    )
                    history_summary = build_history_row(result, self.training_config)
                    history_path, predictions_path = save_run_result(
                        result, self.training_config
                    )
                    duration_seconds = (
                        datetime.now(timezone.utc) - started_at
                    ).total_seconds()
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
                    print(
                        f"Completed {training_task.label} "
                        f"(history: {history_path}, predictions: {predictions_path})"
                    )
                except Exception as error:
                    duration_seconds = (
                        datetime.now(timezone.utc) - started_at
                    ).total_seconds()
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

                if self.store.stop_requested():
                    self.logger.info(
                        "Stop requested after finishing %s. Ending current run.",
                        task_id,
                    )
                    break
        finally:
            self.store.clear_pid_record()
            self._restore_signal_handlers()

        return self.store.summarize()


def launch_background_runner(
    *,
    script_path: Path,
    forwarded_args: list[str],
    cwd: Path,
) -> subprocess.Popen[Any]:
    command = [sys.executable, str(script_path), "run", *forwarded_args]
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
