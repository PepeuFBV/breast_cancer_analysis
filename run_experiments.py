from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from pipeline.config import load_experiment_config
from pipeline.utils.gpu_env import bootstrap_tensorflow_runtime_env
from pipeline.utils.paths import build_project_paths
from pipeline.utils.runtime_limits import CpuExecutionLimits
from pipeline.utils.runtime_probe import collect_runtime_probe, format_runtime_probe, runtime_probe_to_dict
from train import add_training_runtime_arguments

PROJECT_ROOT = Path(__file__).resolve().parent

bootstrap_tensorflow_runtime_env()


def run_one_experiment_task(**kwargs):
    from pipeline.experiments import run_one_experiment_task as _run_one_experiment_task

    return _run_one_experiment_task(**kwargs)


def _add_resolution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help=("Path to an experiment JSON config. " "Defaults to configs/experiment.default.json."),
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Artifacts root directory. Defaults to artifacts/.",
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    add_training_runtime_arguments(parser)
    _add_cpu_limit_arguments(parser)
    parser.add_argument(
        "--isolate-tasks",
        dest="isolate_tasks",
        action="store_true",
        default=None,
        help="Run each selected task in a separate python subprocess.",
    )
    parser.add_argument(
        "--no-isolate-tasks",
        dest="isolate_tasks",
        action="store_false",
        help="Run tasks in-process (legacy behavior).",
    )
    parser.add_argument(
        "--task-cooldown-seconds",
        type=float,
        default=None,
        help="Cooldown before each task execution. Defaults to config runner.task_cooldown_seconds (0).",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout for each isolated task subprocess.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Re-execute experiments previously marked as failed.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Re-execute experiments already completed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many runnable experiments to execute.",
    )
    parser.add_argument(
        "--device-policy",
        choices=["gpu-first", "cpu-only", "gpu-only", "adaptive"],
        default=None,
        help="Execution policy for task device attempts. Defaults to config runner.device_policy (adaptive).",
    )
    parser.add_argument(
        "--gpu-retries",
        type=int,
        default=None,
        help="GPU retry count after OOM before fallback/stop. Defaults to config runner.gpu_retries (1).",
    )
    parser.add_argument(
        "--cpu-retries",
        type=int,
        default=None,
        help="CPU retry count after CPU OOM during fallback. Defaults to config runner.cpu_retries (1).",
    )
    parser.add_argument(
        "--cooldown-after-oom-seconds",
        type=float,
        default=None,
        help="Cooldown after OOM before retrying. Defaults to config runner.cooldown_after_oom_seconds (15).",
    )
    parser.add_argument(
        "--gpu-recovery-cooldown-seconds",
        type=float,
        default=None,
        help="Cooldown before probing GPU again after CPU fallback. Defaults to config runner.gpu_recovery_cooldown_seconds (60).",
    )
    parser.add_argument(
        "--max-consecutive-oom",
        type=int,
        default=None,
        help="Stop the run when final consecutive OOM failures reach this value. Defaults to config runner.max_consecutive_oom (3).",
    )
    parser.add_argument(
        "--max-task-attempts",
        type=int,
        default=None,
        help="Global cap of attempts for a single task across devices. Defaults to config runner.max_task_attempts (4).",
    )
    parser.add_argument(
        "--fail-fast-on-oom",
        action="store_true",
        help="Stop retries/fallback for current task immediately when an OOM is detected.",
    )
    parser.add_argument(
        "--thermal-policy-enabled",
        action="store_true",
        help="Enable thermal-aware runtime controls (CPU/GPU thresholds and cooldowns).",
    )
    parser.add_argument(
        "--thermal-cpu-temp-celsius-limit",
        type=float,
        default=None,
        help="Mark CPU as hot when its temperature reaches this threshold.",
    )
    parser.add_argument(
        "--thermal-cpu-load-percent-limit",
        type=float,
        default=None,
        help="Mark CPU as hot when estimated CPU load reaches this threshold.",
    )
    parser.add_argument(
        "--thermal-gpu-temp-celsius-limit",
        type=float,
        default=None,
        help="Mark GPU as hot when temperature reaches this threshold.",
    )
    parser.add_argument(
        "--thermal-gpu-utilization-percent-limit",
        type=float,
        default=None,
        help="Mark GPU as hot when utilization reaches this threshold.",
    )
    parser.add_argument(
        "--thermal-gpu-recovery-temp-celsius",
        type=float,
        default=None,
        help="Allow GPU return only when temperature is at or below this threshold.",
    )
    parser.add_argument(
        "--thermal-cooldown-seconds",
        type=float,
        default=None,
        help="Pause duration applied by thermal hot-state policies.",
    )
    parser.add_argument(
        "--max-queue-tasks",
        type=int,
        default=None,
        help="Maximum allowed queue size before refusing to run. Defaults to safety limit (50,000).",
    )
    parser.add_argument(
        "--allow-huge-queue",
        action="store_true",
        help="Disable queue-size cap. Use carefully for very large grids.",
    )
    parser.add_argument(
        "--queue-export-path",
        default=None,
        help="Optional path to save the resolved queue as JSONL (metadata + one task per line).",
    )


def _add_cpu_limit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cpu-max-threads",
        type=int,
        default=None,
        help="Cap BLAS/OpenMP worker threads for CPU fallback attempts.",
    )
    parser.add_argument(
        "--cpu-opencv-threads",
        type=int,
        default=None,
        help="Cap OpenCV worker threads for CPU fallback attempts.",
    )
    parser.add_argument(
        "--cpu-inter-op-threads",
        type=int,
        default=None,
        help="Set TF inter-op thread count for CPU fallback attempts.",
    )
    parser.add_argument(
        "--cpu-intra-op-threads",
        type=int,
        default=None,
        help="Set TF intra-op thread count for CPU fallback attempts.",
    )
    parser.add_argument(
        "--cpu-nice",
        type=int,
        default=None,
        help="Apply a positive Linux nice level to CPU fallback attempts when supported.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Control iterative, resumable execution of training experiments."))
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser(
        "probe-runtime",
        help="Inspect TensorFlow/runtime device visibility and thread limits.",
    )
    probe_parser.add_argument(
        "--config",
        default=None,
        help=("Path to an experiment JSON config. " "Defaults to configs/experiment.default.json."),
    )
    probe_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Runtime device to probe.",
    )
    probe_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the runtime probe as JSON.",
    )
    _add_cpu_limit_arguments(probe_parser)

    count_parser = subparsers.add_parser(
        "count",
        help="Dry run: print queue dimensions, total experiments, and total fits.",
    )
    add_training_runtime_arguments(count_parser)
    count_parser.add_argument(
        "--json",
        action="store_true",
        help="Print count output as JSON.",
    )

    dry_run_parser = subparsers.add_parser(
        "dry-run",
        help="Alias for `count`.",
    )
    add_training_runtime_arguments(dry_run_parser)
    dry_run_parser.add_argument(
        "--json",
        action="store_true",
        help="Print count output as JSON.",
    )

    run_task_parser = subparsers.add_parser(
        "run-task",
        help="Run exactly one task from the persisted experiment queue.",
    )
    add_training_runtime_arguments(run_task_parser)
    _add_cpu_limit_arguments(run_task_parser)
    run_task_parser.add_argument(
        "--task-cooldown-seconds",
        type=float,
        default=None,
        help="Cooldown before executing the task. Defaults to config runner.task_cooldown_seconds (0).",
    )
    run_task_parser.add_argument(
        "--task-id",
        required=True,
        help="Persisted experiment task id (example: exp-xxxxxxxxxxxxxxxx).",
    )
    run_task_parser.add_argument(
        "--task-record-json",
        default=None,
        help="Serialized task record used by isolated child workers to avoid rebuilding the full queue.",
    )
    run_task_parser.add_argument(
        "--max-queue-tasks",
        type=int,
        default=None,
        help="Maximum allowed queue size before refusing to run. Defaults to safety limit (50,000).",
    )
    run_task_parser.add_argument(
        "--allow-huge-queue",
        action="store_true",
        help="Disable queue-size cap. Use carefully for very large grids.",
    )
    run_task_parser.add_argument(
        "--queue-export-path",
        default=None,
        help="Optional path to save the resolved queue as JSONL (metadata + one task per line).",
    )

    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch the iterative runner in the background.",
    )
    _add_run_arguments(launch_parser)
    launch_parser.add_argument(
        "--_launch-worker",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    stop_parser = subparsers.add_parser(
        "stop",
        help="Request a graceful stop after the current experiment.",
    )
    _add_resolution_arguments(stop_parser)
    stop_parser.add_argument(
        "--kill",
        action="store_true",
        help="Immediately terminate the active runner process.",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Show the current execution summary.",
    )
    _add_resolution_arguments(status_parser)
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the status snapshot as JSON.",
    )

    partial_parser = subparsers.add_parser(
        "partial",
        help="Extract completed-results-so-far from persisted runner state.",
    )
    _add_resolution_arguments(partial_parser)
    partial_parser.add_argument(
        "--json",
        action="store_true",
        help="Print partial results as JSON (default behavior).",
    )
    partial_parser.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="Maximum number of finalized task rows to include (default: 500).",
    )
    partial_parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Disable row cap and include all finalized rows.",
    )
    partial_parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional CSV file path to write extracted task rows.",
    )

    reset_parser = subparsers.add_parser(
        "reset",
        help="Reset persisted orchestration state.",
    )
    _add_resolution_arguments(reset_parser)
    reset_parser.add_argument(
        "--purge-results",
        action="store_true",
        help="Also remove saved training history, predictions and runner metadata.",
    )
    reset_parser.add_argument(
        "--kill-active",
        action="store_true",
        help="Immediately terminate an active runner before resetting state.",
    )

    return parser


def _load_project_paths(args: argparse.Namespace):
    experiment_config = load_experiment_config(args.config)
    return experiment_config.resolve_project_paths(artifacts_dir=args.artifacts_dir).ensure_artifact_dirs()


def _resolve_cpu_execution_limits(args: argparse.Namespace) -> CpuExecutionLimits:
    experiment_config = load_experiment_config(getattr(args, "config", None))
    return experiment_config.build_cpu_execution_limits(
        cpu_max_threads=getattr(args, "cpu_max_threads", None),
        cpu_opencv_threads=getattr(args, "cpu_opencv_threads", None),
        cpu_inter_op_threads=getattr(args, "cpu_inter_op_threads", None),
        cpu_intra_op_threads=getattr(args, "cpu_intra_op_threads", None),
        cpu_nice=getattr(args, "cpu_nice", None),
    )


def _build_runner(args: argparse.Namespace):
    from pipeline.experiments import IterativeExperimentRunner

    experiment_config = load_experiment_config(args.config)
    project_paths = experiment_config.resolve_project_paths(
        raw_data_dir=args.raw_data_dir,
        artifacts_dir=args.artifacts_dir,
    ).ensure_artifact_dirs()
    training_config = experiment_config.build_training_config(
        project_paths,
        train_split=args.train_split,
        test_split=args.test_split,
        history_dir=args.history_dir,
        predictions_dir=args.predictions_dir,
        folds=args.folds,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        loss=args.loss,
        random_state=args.random_state,
        model_names=args.models,
        preprocessing_ids=args.preprocessing,
        include_combinations=args.include_combinations,
        augmentations_per_image=args.augmentations_per_image,
        run_skip=args.run_skip,
    )
    return IterativeExperimentRunner(
        config_path=experiment_config.source_path,
        project_paths=project_paths,
        training_config=training_config,
        cpu_execution_limits=_resolve_cpu_execution_limits(args),
    )


def _resolve_runner_cli_options(
    args: argparse.Namespace,
) -> tuple[bool, float, float | None, str, int, int, float, float, int, int, bool, bool, float | None, float | None, float | None, float | None, float | None, float]:
    experiment_config = load_experiment_config(args.config)
    isolate_tasks = experiment_config.runner.isolate_tasks if getattr(args, "isolate_tasks", None) is None else bool(getattr(args, "isolate_tasks"))
    cooldown_seconds = experiment_config.runner.task_cooldown_seconds if getattr(args, "task_cooldown_seconds", None) is None else float(getattr(args, "task_cooldown_seconds"))
    if cooldown_seconds < 0:
        raise ValueError("--task-cooldown-seconds must be >= 0.")
    timeout_seconds = experiment_config.runner.task_timeout_seconds if getattr(args, "task_timeout_seconds", None) is None else float(getattr(args, "task_timeout_seconds"))
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("--task-timeout-seconds must be > 0 when provided.")
    config_device_policy = getattr(experiment_config.runner, "device_policy", "adaptive")
    config_gpu_retries = int(getattr(experiment_config.runner, "gpu_retries", 1))
    config_cpu_retries = int(getattr(experiment_config.runner, "cpu_retries", 1))
    config_cooldown_after_oom = float(getattr(experiment_config.runner, "cooldown_after_oom_seconds", 15.0))
    config_gpu_recovery_cooldown = float(getattr(experiment_config.runner, "gpu_recovery_cooldown_seconds", 60.0))
    config_max_consecutive_oom = int(getattr(experiment_config.runner, "max_consecutive_oom", 3))
    config_max_task_attempts = int(getattr(experiment_config.runner, "max_task_attempts", 4))
    config_fail_fast_on_oom = bool(getattr(experiment_config.runner, "fail_fast_on_oom", False))
    config_thermal_policy_enabled = bool(getattr(experiment_config.runner, "thermal_policy_enabled", False))
    config_thermal_cpu_temp_celsius_limit = getattr(experiment_config.runner, "thermal_cpu_temp_celsius_limit", None)
    config_thermal_cpu_load_percent_limit = getattr(experiment_config.runner, "thermal_cpu_load_percent_limit", None)
    config_thermal_gpu_temp_celsius_limit = getattr(experiment_config.runner, "thermal_gpu_temp_celsius_limit", None)
    config_thermal_gpu_utilization_percent_limit = getattr(experiment_config.runner, "thermal_gpu_utilization_percent_limit", None)
    config_thermal_gpu_recovery_temp_celsius = getattr(experiment_config.runner, "thermal_gpu_recovery_temp_celsius", None)
    config_thermal_cooldown_seconds = float(getattr(experiment_config.runner, "thermal_cooldown_seconds", 30.0))

    device_policy = config_device_policy if getattr(args, "device_policy", None) is None else str(getattr(args, "device_policy"))
    gpu_retries = config_gpu_retries if getattr(args, "gpu_retries", None) is None else int(getattr(args, "gpu_retries"))
    cpu_retries = config_cpu_retries if getattr(args, "cpu_retries", None) is None else int(getattr(args, "cpu_retries"))
    cooldown_after_oom_seconds = config_cooldown_after_oom if getattr(args, "cooldown_after_oom_seconds", None) is None else float(getattr(args, "cooldown_after_oom_seconds"))
    gpu_recovery_cooldown_seconds = config_gpu_recovery_cooldown if getattr(args, "gpu_recovery_cooldown_seconds", None) is None else float(getattr(args, "gpu_recovery_cooldown_seconds"))
    max_consecutive_oom = config_max_consecutive_oom if getattr(args, "max_consecutive_oom", None) is None else int(getattr(args, "max_consecutive_oom"))
    max_task_attempts = config_max_task_attempts if getattr(args, "max_task_attempts", None) is None else int(getattr(args, "max_task_attempts"))
    fail_fast_on_oom = bool(getattr(args, "fail_fast_on_oom", False) or config_fail_fast_on_oom)
    thermal_policy_enabled = bool(getattr(args, "thermal_policy_enabled", False) or config_thermal_policy_enabled)
    thermal_cpu_temp_celsius_limit = (
        config_thermal_cpu_temp_celsius_limit
        if getattr(args, "thermal_cpu_temp_celsius_limit", None) is None
        else float(getattr(args, "thermal_cpu_temp_celsius_limit"))
    )
    thermal_cpu_load_percent_limit = (
        config_thermal_cpu_load_percent_limit
        if getattr(args, "thermal_cpu_load_percent_limit", None) is None
        else float(getattr(args, "thermal_cpu_load_percent_limit"))
    )
    thermal_gpu_temp_celsius_limit = (
        config_thermal_gpu_temp_celsius_limit
        if getattr(args, "thermal_gpu_temp_celsius_limit", None) is None
        else float(getattr(args, "thermal_gpu_temp_celsius_limit"))
    )
    thermal_gpu_utilization_percent_limit = (
        config_thermal_gpu_utilization_percent_limit
        if getattr(args, "thermal_gpu_utilization_percent_limit", None) is None
        else float(getattr(args, "thermal_gpu_utilization_percent_limit"))
    )
    thermal_gpu_recovery_temp_celsius = (
        config_thermal_gpu_recovery_temp_celsius
        if getattr(args, "thermal_gpu_recovery_temp_celsius", None) is None
        else float(getattr(args, "thermal_gpu_recovery_temp_celsius"))
    )
    thermal_cooldown_seconds = (
        config_thermal_cooldown_seconds
        if getattr(args, "thermal_cooldown_seconds", None) is None
        else float(getattr(args, "thermal_cooldown_seconds"))
    )
    if thermal_cooldown_seconds < 0:
        raise ValueError("--thermal-cooldown-seconds must be >= 0.")
    if thermal_cpu_temp_celsius_limit is not None and thermal_cpu_temp_celsius_limit <= 0:
        raise ValueError("--thermal-cpu-temp-celsius-limit must be > 0.")
    if thermal_gpu_temp_celsius_limit is not None and thermal_gpu_temp_celsius_limit <= 0:
        raise ValueError("--thermal-gpu-temp-celsius-limit must be > 0.")
    if thermal_gpu_recovery_temp_celsius is not None and thermal_gpu_recovery_temp_celsius <= 0:
        raise ValueError("--thermal-gpu-recovery-temp-celsius must be > 0.")
    if thermal_cpu_load_percent_limit is not None and not 0 <= thermal_cpu_load_percent_limit <= 100:
        raise ValueError("--thermal-cpu-load-percent-limit must be between 0 and 100.")
    if thermal_gpu_utilization_percent_limit is not None and not 0 <= thermal_gpu_utilization_percent_limit <= 100:
        raise ValueError("--thermal-gpu-utilization-percent-limit must be between 0 and 100.")
    if (
        thermal_gpu_temp_celsius_limit is not None
        and thermal_gpu_recovery_temp_celsius is not None
        and thermal_gpu_recovery_temp_celsius > thermal_gpu_temp_celsius_limit
    ):
        raise ValueError("--thermal-gpu-recovery-temp-celsius must be <= --thermal-gpu-temp-celsius-limit.")
    return (
        isolate_tasks,
        cooldown_seconds,
        timeout_seconds,
        device_policy,
        gpu_retries,
        cpu_retries,
        cooldown_after_oom_seconds,
        gpu_recovery_cooldown_seconds,
        max_consecutive_oom,
        max_task_attempts,
        fail_fast_on_oom,
        thermal_policy_enabled,
        thermal_cpu_temp_celsius_limit,
        thermal_cpu_load_percent_limit,
        thermal_gpu_temp_celsius_limit,
        thermal_gpu_utilization_percent_limit,
        thermal_gpu_recovery_temp_celsius,
        thermal_cooldown_seconds,
    )


def _build_run_task_forwarded_args(
    args: argparse.Namespace,
    *,
    task_cooldown_seconds: float,
) -> list[str]:
    forwarded: list[str] = []

    def _add_optional(name: str, value: object) -> None:
        if value is None:
            return
        forwarded.extend([name, str(value)])

    def _add_optional_many(name: str, values: list[str] | None) -> None:
        if not values:
            return
        forwarded.append(name)
        forwarded.extend(values)

    _add_optional("--config", args.config)
    _add_optional("--raw-data-dir", args.raw_data_dir)
    _add_optional("--artifacts-dir", args.artifacts_dir)
    _add_optional("--train-split", args.train_split)
    _add_optional("--test-split", args.test_split)
    _add_optional("--history-dir", args.history_dir)
    _add_optional("--predictions-dir", args.predictions_dir)
    _add_optional("--folds", args.folds)
    _add_optional("--validation-size", args.validation_size)
    _add_optional("--random-state", args.random_state)
    _add_optional("--batch-size", args.batch_size)
    _add_optional("--epochs", args.epochs)
    _add_optional("--learning-rate", args.learning_rate)
    _add_optional("--loss", args.loss)
    _add_optional_many("--models", args.models)
    _add_optional_many("--preprocessing", args.preprocessing)
    _add_optional_many(
        "--augmentations-per-image",
        (None if args.augmentations_per_image is None else [str(value) for value in args.augmentations_per_image]),
    )
    _add_optional("--task-cooldown-seconds", task_cooldown_seconds)
    _add_optional("--max-queue-tasks", args.max_queue_tasks)
    _add_optional("--cpu-max-threads", getattr(args, "cpu_max_threads", None))
    _add_optional("--cpu-opencv-threads", getattr(args, "cpu_opencv_threads", None))
    _add_optional("--cpu-inter-op-threads", getattr(args, "cpu_inter_op_threads", None))
    _add_optional("--cpu-intra-op-threads", getattr(args, "cpu_intra_op_threads", None))
    _add_optional("--cpu-nice", getattr(args, "cpu_nice", None))
    if bool(getattr(args, "allow_huge_queue", False)):
        forwarded.append("--allow-huge-queue")
    if args.include_combinations is not None:
        forwarded.append("--combined-preprocessing" if bool(args.include_combinations) else "--no-combined-preprocessing")
    if args.run_skip is not None:
        forwarded.append("--run-skip" if bool(args.run_skip) else "--no-run-skip")

    return forwarded


def _resolve_max_queue_tasks(args: argparse.Namespace) -> int | None:
    max_queue_tasks = getattr(args, "max_queue_tasks", None)
    allow_huge_queue = bool(getattr(args, "allow_huge_queue", False))
    if allow_huge_queue:
        return None
    return max_queue_tasks


def _resolution_args_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if getattr(args, "config", None):
        parts.extend(["--config", str(args.config)])
    if getattr(args, "artifacts_dir", None):
        parts.extend(["--artifacts-dir", str(args.artifacts_dir)])
    return "" if not parts else " " + " ".join(parts)


def _discover_active_runner_stores(*, exclude_artifacts_dir: Path | None = None) -> list[dict[str, Any]]:
    from pipeline.experiments import ExperimentStateStore

    discovered: list[dict[str, Any]] = []
    seen_artifacts_dirs: set[Path] = set()
    excluded = None if exclude_artifacts_dir is None else exclude_artifacts_dir.resolve()
    for pid_path in sorted(PROJECT_ROOT.glob("artifacts*/experiments/control/runner_pid.json")):
        artifacts_dir = pid_path.parents[2]
        resolved_artifacts_dir = artifacts_dir.resolve()
        if excluded is not None and resolved_artifacts_dir == excluded:
            continue
        if resolved_artifacts_dir in seen_artifacts_dirs:
            continue
        seen_artifacts_dirs.add(resolved_artifacts_dir)
        store = ExperimentStateStore(build_project_paths(artifacts_dir=artifacts_dir))
        if not store.has_active_run():
            continue
        pid_record = store.read_pid_record() or {}
        discovered.append(
            {
                "store": store,
                "artifacts_dir": artifacts_dir,
                "pid": pid_record.get("pid"),
                "config_path": pid_record.get("config_path"),
            }
        )
    return discovered


def _resolve_store_for_control_command(args: argparse.Namespace):
    from pipeline.experiments import ExperimentStateStore

    project_paths = _load_project_paths(args)
    default_store = ExperimentStateStore(project_paths)
    if getattr(args, "config", None) is not None or getattr(args, "artifacts_dir", None) is not None:
        return default_store, None
    if default_store.has_active_run():
        return default_store, None

    discovered = _discover_active_runner_stores(exclude_artifacts_dir=project_paths.artifacts_dir)
    if len(discovered) == 1:
        match = discovered[0]
        return (
            match["store"],
            "Using the only active runner found under " f"{match['artifacts_dir']} (config: {match['config_path'] or 'unknown'}).",
        )
    if len(discovered) > 1:
        details = "\n".join([f" - pid={match['pid']} artifacts={match['artifacts_dir']} " f"config={match['config_path'] or 'unknown'}" for match in discovered])
        raise RuntimeError("Multiple active runners were found. Use --config or --artifacts-dir " f"to choose one:\n{details}")
    return default_store, None


def _print_status_snapshot(snapshot: dict[str, object]) -> None:
    counts = snapshot["counts"]
    current_task = snapshot["current_task"]
    if snapshot.get("selection_note"):
        print(str(snapshot["selection_note"]))
    print(f"Overall status: {snapshot['overall_status']}")
    print(f"Total experiments: {snapshot['total']}")
    print(f"Completed: {counts['completed']}")
    print(f"Failed: {counts['failed']}")
    print(f"Pending: {counts['pending']}")
    print(f"Stopped: {counts['stopped']}")
    if snapshot["active_pid"] is not None:
        print(f"Active PID: {snapshot['active_pid']}")
    if snapshot.get("background_stdout_log_path"):
        print(f"Background stdout log: {snapshot['background_stdout_log_path']}")
    if snapshot.get("background_stderr_log_path"):
        print(f"Background stderr log: {snapshot['background_stderr_log_path']}")
    if snapshot.get("desired_state") is not None:
        print(f"Desired state: {snapshot['desired_state']}")
    if snapshot.get("pause_reason"):
        print(f"Pause reason: {snapshot['pause_reason']}")
    if snapshot.get("pause_requested_at"):
        print(f"Pause requested at: {snapshot['pause_requested_at']}")
    if snapshot.get("resume_requested_at"):
        print(f"Resume requested at: {snapshot['resume_requested_at']}")
    if snapshot.get("last_recovery_reason"):
        print(f"Last recovery: {snapshot['last_recovery_reason']}")
    if snapshot.get("last_recovery_at"):
        print(f"Last recovery at: {snapshot['last_recovery_at']}")
    if current_task:
        print("Current task: " f"{current_task['preproc_id']} [{current_task['model_name']} - " f"{current_task['param_display']}]")
    if snapshot.get("device_policy") is not None:
        print(f"Device policy: {snapshot['device_policy']}")
    if snapshot.get("isolate_tasks") is not None:
        print(f"Isolate tasks: {snapshot['isolate_tasks']}")
    if snapshot.get("allow_huge_queue") is not None:
        print(f"Allow huge queue: {snapshot['allow_huge_queue']}")
    if snapshot.get("max_queue_tasks") is not None:
        print(f"Max queue tasks: {snapshot['max_queue_tasks']}")
    if snapshot.get("stream_queue_mode") is not None:
        print(f"Stream queue mode: {snapshot['stream_queue_mode']}")
    if snapshot.get("preferred_device") is not None:
        print(f"Current preferred device: {snapshot['preferred_device']}")
    if snapshot.get("gpu_health") is not None:
        print(f"GPU health: {snapshot['gpu_health']}")
    if snapshot.get("last_gpu_oom_task_id"):
        print(f"Last GPU OOM task: {snapshot['last_gpu_oom_task_id']}")
    if snapshot.get("gpu_oom_count") is not None:
        print(f"GPU OOM count: {snapshot['gpu_oom_count']}")
    if snapshot.get("cpu_fallback_successes") is not None:
        print(f"CPU fallback successes: {snapshot['cpu_fallback_successes']}")
    if snapshot.get("consecutive_final_oom_failures") is not None:
        print("Consecutive final OOM failures: " f"{snapshot['consecutive_final_oom_failures']}")
    if snapshot.get("last_successful_device") is not None:
        print(f"Last successful device: {snapshot['last_successful_device']}")
    if snapshot.get("oom_policy_stop"):
        print(f"OOM policy stop: {snapshot['oom_policy_stop']}")
    if snapshot.get("thermal_policy_enabled") is not None:
        print(f"Thermal policy enabled: {snapshot['thermal_policy_enabled']}")
    if snapshot.get("thermal_state") is not None:
        print(f"Thermal state: {snapshot['thermal_state']}")
    if snapshot.get("thermal_last_reason"):
        print(f"Thermal reason: {snapshot['thermal_last_reason']}")
    if snapshot.get("thermal_last_sample_at"):
        print(f"Thermal sample at: {snapshot['thermal_last_sample_at']}")
    if snapshot.get("thermal_last_cpu_temp_celsius") is not None:
        print("Thermal CPU temp (C): " f"{snapshot['thermal_last_cpu_temp_celsius']}")
    if snapshot.get("thermal_last_cpu_load_percent") is not None:
        print("Thermal CPU load (%): " f"{snapshot['thermal_last_cpu_load_percent']}")
    if snapshot.get("thermal_last_gpu_temp_celsius") is not None:
        print("Thermal GPU temp (C): " f"{snapshot['thermal_last_gpu_temp_celsius']}")
    if snapshot.get("thermal_last_gpu_utilization_percent") is not None:
        print("Thermal GPU util (%): " f"{snapshot['thermal_last_gpu_utilization_percent']}")
    if snapshot.get("config_path"):
        print(f"Config: {snapshot['config_path']}")
    if snapshot.get("results_root"):
        print(f"Artifacts root: {snapshot['results_root']}")
    print(f"State file: {snapshot['state_path']}")
    print(f"Summary file: {snapshot['summary_path']}")
    print(f"History dir: {snapshot['history_dir']}")
    print(f"Predictions dir: {snapshot['predictions_dir']}")
    print(f"Runner log: {snapshot['log_path']}")
    state_path = Path(str(snapshot["state_path"]))
    if snapshot["total"] == 0 and not state_path.exists():
        print("Note: no persisted runner state exists yet.")


def _write_partial_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "status"])
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_runtime_probe(args: argparse.Namespace) -> int:
    probe_result = collect_runtime_probe(
        device=args.device,
        cpu_execution_limits=_resolve_cpu_execution_limits(args),
    )
    if args.json:
        print(json.dumps(runtime_probe_to_dict(probe_result), indent=2, sort_keys=True))
    else:
        print(format_runtime_probe(probe_result))
    return 0 if probe_result.ok else 1


def _count_payload(args: argparse.Namespace) -> dict[str, int]:
    runner = _build_runner(args)
    counts = runner.estimate_grid_counts()
    return {
        "preprocessing_count": int(counts.preprocessing_count),
        "model_count": int(counts.model_count),
        "augmentation_count": int(counts.augmentation_count),
        "total_experiments": int(counts.total_experiments),
        "folds": int(counts.folds),
        "total_fits": int(counts.total_fits),
    }


def _print_count_result(args: argparse.Namespace) -> int:
    payload = _count_payload(args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print("Experiment count (dry run):")
    print(f"Preprocessing variants: {payload['preprocessing_count']:,}")
    print(f"Models: {payload['model_count']:,}")
    print(f"Augmentation values: {payload['augmentation_count']:,}")
    print(f"Total experiments: {payload['total_experiments']:,}")
    print(f"Folds: {payload['folds']:,}")
    print(f"Total fits: {payload['total_fits']:,}")
    return 0


def _format_launch_startup_estimate(*, total_experiments: int, stream_queue_mode: bool) -> str:
    if stream_queue_mode:
        if total_experiments >= 1_000_000:
            return "about 10-45 seconds"
        if total_experiments >= 250_000:
            return "about 10-30 seconds"
        return "about 5-20 seconds"
    if total_experiments >= 75_000:
        return "about 1-5 minutes"
    if total_experiments >= 25_000:
        return "about 20-90 seconds"
    return "about 5-20 seconds"


def _resolve_launch_preflight(args: argparse.Namespace) -> dict[str, Any]:
    from pipeline.experiments.runner import STREAMING_QUEUE_TASK_THRESHOLD

    runner = _build_runner(args)
    counts = runner.estimate_grid_counts()
    (
        isolate_tasks,
        _task_cooldown_seconds,
        _task_timeout_seconds,
        device_policy,
        _gpu_retries,
        _cpu_retries,
        _cooldown_after_oom_seconds,
        _gpu_recovery_cooldown_seconds,
        _max_consecutive_oom,
        _max_task_attempts,
        _fail_fast_on_oom,
        _thermal_policy_enabled,
        _thermal_cpu_temp_celsius_limit,
        _thermal_cpu_load_percent_limit,
        _thermal_gpu_temp_celsius_limit,
        _thermal_gpu_utilization_percent_limit,
        _thermal_gpu_recovery_temp_celsius,
        _thermal_cooldown_seconds,
    ) = _resolve_runner_cli_options(args)
    has_process_local_components = runner.model_builders is not None or runner.preprocessing_tasks is not None
    max_queue_tasks = _resolve_max_queue_tasks(args)
    stream_queue_mode = max_queue_tasks is None and counts.total_experiments > STREAMING_QUEUE_TASK_THRESHOLD and not has_process_local_components
    if stream_queue_mode and (args.rerun_failed or args.rerun_completed):
        raise ValueError("rerun_failed/rerun_completed are not supported with streamed huge-queue execution. " "Use a targeted rerun instead.")
    return {
        "counts": counts,
        "isolate_tasks": isolate_tasks,
        "device_policy": device_policy,
        "stream_queue_mode": stream_queue_mode,
        "max_queue_tasks": max_queue_tasks,
        "startup_estimate": _format_launch_startup_estimate(
            total_experiments=int(counts.total_experiments),
            stream_queue_mode=stream_queue_mode,
        ),
    }


def _print_launch_preflight(preflight: dict[str, Any]) -> None:
    counts = preflight["counts"]
    queue_mode = "streamed" if preflight["stream_queue_mode"] else "materialized"
    print("Launch preflight:")
    print(f"Queue mode: {queue_mode}")
    print(f"Estimated startup overhead: {preflight['startup_estimate']} (heuristic)")
    print(f"Total experiments: {int(counts.total_experiments):,}")
    print(f"Total fits: {int(counts.total_fits):,}")
    print(f"Device policy: {preflight['device_policy']}")
    print(f"Isolate tasks: {preflight['isolate_tasks']}")
    if preflight["max_queue_tasks"] is None:
        print("Max queue tasks: unlimited")
    else:
        print(f"Max queue tasks: {int(preflight['max_queue_tasks']):,}")
    if preflight["stream_queue_mode"]:
        print("Warning: huge queue detected; the runner will stream task discovery instead of materializing full state.")
    elif int(counts.total_experiments) >= 25_000:
        print("Warning: large materialized queue; first task may be delayed while state is written.")


def main(argv: list[str] | None = None) -> int:
    resolved_argv = sys.argv[1:] if argv is None else argv
    if resolved_argv and resolved_argv[0] == "run":
        print("The `run` command was removed. Use `python run_experiments.py launch ...` instead.")
        return 2
    args = build_parser().parse_args(resolved_argv)

    if args.command == "probe-runtime":
        return _print_runtime_probe(args)

    if args.command in {"count", "dry-run"}:
        try:
            return _print_count_result(args)
        except ValueError as error:
            print(str(error))
            return 1

    if args.command == "launch":
        from pipeline.experiments import ExperimentStateStore, IterativeRunOptions, launch_background_runner

        if bool(getattr(args, "_launch_worker", False)):
            runner = _build_runner(args)
            try:
                (
                    isolate_tasks,
                    task_cooldown_seconds,
                    task_timeout_seconds,
                    device_policy,
                    gpu_retries,
                    cpu_retries,
                    cooldown_after_oom_seconds,
                    gpu_recovery_cooldown_seconds,
                    max_consecutive_oom,
                    max_task_attempts,
                    fail_fast_on_oom,
                    thermal_policy_enabled,
                    thermal_cpu_temp_celsius_limit,
                    thermal_cpu_load_percent_limit,
                    thermal_gpu_temp_celsius_limit,
                    thermal_gpu_utilization_percent_limit,
                    thermal_gpu_recovery_temp_celsius,
                    thermal_cooldown_seconds,
                ) = _resolve_runner_cli_options(args)
                run_task_command_base = (
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "run-task",
                    *_build_run_task_forwarded_args(
                        args,
                        task_cooldown_seconds=task_cooldown_seconds,
                    ),
                )
                snapshot = runner.run(
                    IterativeRunOptions(
                        rerun_failed=args.rerun_failed,
                        rerun_completed=args.rerun_completed,
                        limit=args.limit,
                        isolate_tasks=isolate_tasks,
                        task_cooldown_seconds=task_cooldown_seconds,
                        task_timeout_seconds=task_timeout_seconds,
                        run_task_command_base=run_task_command_base,
                        device_policy=device_policy,
                        gpu_retries=gpu_retries,
                        cpu_retries=cpu_retries,
                        cooldown_after_oom_seconds=cooldown_after_oom_seconds,
                        gpu_recovery_cooldown_seconds=gpu_recovery_cooldown_seconds,
                        max_consecutive_oom=max_consecutive_oom,
                        max_task_attempts=max_task_attempts,
                        fail_fast_on_oom=fail_fast_on_oom,
                        thermal_policy_enabled=thermal_policy_enabled,
                        thermal_cpu_temp_celsius_limit=thermal_cpu_temp_celsius_limit,
                        thermal_cpu_load_percent_limit=thermal_cpu_load_percent_limit,
                        thermal_gpu_temp_celsius_limit=thermal_gpu_temp_celsius_limit,
                        thermal_gpu_utilization_percent_limit=thermal_gpu_utilization_percent_limit,
                        thermal_gpu_recovery_temp_celsius=thermal_gpu_recovery_temp_celsius,
                        thermal_cooldown_seconds=thermal_cooldown_seconds,
                        max_queue_tasks=_resolve_max_queue_tasks(args),
                        queue_export_path=args.queue_export_path,
                    )
                )
            except RuntimeError as error:
                print(str(error))
                return 1
            except ValueError as error:
                print(str(error))
                return 1
            except KeyboardInterrupt:
                snapshot = runner.store.summarize()
                if snapshot["total"] == 0:
                    print("Run interrupted before any experiment state was created.")
                else:
                    print("Run interrupted.")
                    _print_status_snapshot(snapshot)
                return 130
            _print_status_snapshot(snapshot)
            return 0

        project_paths = _load_project_paths(args)
        store = ExperimentStateStore(project_paths)
        if store.has_active_run():
            snapshot = store.summarize()
            pid = snapshot.get("active_pid")
            print(f"Runner already active with pid={pid}.")
            print(f"Check: python run_experiments.py status{_resolution_args_suffix(args)}")
            print(f"Graceful stop: python run_experiments.py stop{_resolution_args_suffix(args)}")
            print(f"Immediate stop: python run_experiments.py stop{_resolution_args_suffix(args)} --kill")
            return 1
        try:
            preflight = _resolve_launch_preflight(args)
        except ValueError as error:
            print(str(error))
            return 1
        _print_launch_preflight(preflight)
        process = launch_background_runner(
            script_path=Path(__file__).resolve(),
            forwarded_args=resolved_argv[1:],
            cwd=PROJECT_ROOT,
            logs_dir=project_paths.experiment_logs_dir,
        )
        store.mark_launch_requested()
        print(f"Background runner started with pid={process.process.pid}.")
        print(f"stdout: {process.stdout_path}")
        print(f"stderr: {process.stderr_path}")
        print(f"Check status with: python run_experiments.py status{_resolution_args_suffix(args)}")
        return 0

    if args.command == "run-task":
        runner = _build_runner(args)
        try:
            _, task_cooldown_seconds, _, *_unused = _resolve_runner_cli_options(args)
            result = run_one_experiment_task(
                task_id=args.task_id,
                config_path=runner.config_path,
                project_paths=runner.project_paths,
                training_config=runner.training_config,
                task_cooldown_seconds=task_cooldown_seconds,
                max_queue_tasks=_resolve_max_queue_tasks(args),
                queue_export_path=args.queue_export_path,
                task_record=(None if args.task_record_json is None else json.loads(args.task_record_json)),
                cpu_execution_limits=_resolve_cpu_execution_limits(args),
            )
        except RuntimeError as error:
            print(str(error))
            return 1
        except ValueError as error:
            print(str(error))
            return 1
        except KeyError as error:
            print(str(error))
            return 1
        except KeyboardInterrupt:
            print("Single-task run interrupted.")
            return 130

        status = str(result.get("status", "failed"))
        if status == "completed":
            if result.get("skipped"):
                print("Task marked as completed without re-execution " f"(reason: {result.get('skip_reason')}).")
            else:
                print(f"Task {args.task_id} completed successfully.")
            return 0

        error_summary = result.get("error_summary")
        if error_summary:
            print(f"Task {args.task_id} failed: {error_summary}")
        else:
            print(f"Task {args.task_id} failed with status={status}.")
        return 1

    try:
        store, selection_note = _resolve_store_for_control_command(args)
    except RuntimeError as error:
        print(str(error))
        return 1

    if args.command == "stop":
        if selection_note:
            print(selection_note)
        if getattr(args, "kill", False):
            try:
                if not store.has_active_run():
                    print("No active runner process found.")
                    return 0
                pid = store.terminate_active_run(force=True)
            except RuntimeError as error:
                print(str(error))
                return 1
            print(f"Runner pid={pid} was terminated.")
            return 0
        stop_path = store.request_stop()
        if not store.has_active_run():
            print("No active runner process found.")
            print("Pause was persisted and will be honored until the next `launch` request.")
            print(f"Stop flag: {stop_path}")
            return 0
        print("Stop requested. The runner will finish the current experiment and stop.")
        print(f"Stop flag: {stop_path}")
        return 0

    if args.command == "status":
        snapshot = store.summarize()
        if selection_note:
            snapshot["selection_note"] = selection_note
        if args.json:
            print(json.dumps(snapshot, indent=2, sort_keys=True))
        else:
            _print_status_snapshot(snapshot)
        return 0

    if args.command == "partial":
        try:
            payload = store.partial_results(
                max_rows=args.max_rows,
                all_rows=bool(args.all_rows),
            )
        except ValueError as error:
            print(str(error))
            return 1
        if selection_note:
            payload["selection_note"] = selection_note
        if args.csv_output:
            csv_path = Path(args.csv_output).expanduser()
            _write_partial_rows_csv(csv_path, list(payload.get("partial_rows", [])))
            print(f"Partial rows written to: {csv_path}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "reset":
        if selection_note:
            print(selection_note)
        try:
            store.reset(
                purge_results=args.purge_results,
                kill_active=bool(getattr(args, "kill_active", False)),
            )
        except PermissionError as error:
            print("Reset failed because a file is still in use: " f"{error.filename or str(error)}")
            return 1
        except RuntimeError as error:
            print(str(error))
            return 1
        if args.purge_results:
            print("Runner state and training results were removed.")
        else:
            print("Runner state was reset. Existing training artifacts were preserved.")
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
