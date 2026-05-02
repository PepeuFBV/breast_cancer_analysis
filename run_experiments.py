from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.config import load_experiment_config
from pipeline.experiments import (
    ExperimentStateStore,
    IterativeExperimentRunner,
    IterativeRunOptions,
    launch_background_runner,
    run_one_experiment_task,
)
from pipeline.utils.gpu_env import ensure_tensorflow_wsl_gpu_env
from train import add_training_runtime_arguments

PROJECT_ROOT = Path(__file__).resolve().parent

ensure_tensorflow_wsl_gpu_env()


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
        help="Cooldown before each task execution. Defaults to config runner.task_cooldown_seconds (2).",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Control iterative, resumable execution of training experiments."))
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run or resume experiments.")
    _add_run_arguments(run_parser)

    run_task_parser = subparsers.add_parser(
        "run-task",
        help="Run exactly one task from the persisted experiment queue.",
    )
    add_training_runtime_arguments(run_task_parser)
    run_task_parser.add_argument(
        "--task-cooldown-seconds",
        type=float,
        default=None,
        help="Cooldown before executing the task. Defaults to config runner.task_cooldown_seconds (2).",
    )
    run_task_parser.add_argument(
        "--task-id",
        required=True,
        help="Persisted experiment task id (example: exp-xxxxxxxxxxxxxxxx).",
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

    stop_parser = subparsers.add_parser(
        "stop",
        help="Request a graceful stop after the current experiment.",
    )
    _add_resolution_arguments(stop_parser)

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

    return parser


def _load_project_paths(args: argparse.Namespace):
    experiment_config = load_experiment_config(args.config)
    return experiment_config.resolve_project_paths(artifacts_dir=args.artifacts_dir).ensure_artifact_dirs()


def _build_runner(args: argparse.Namespace) -> IterativeExperimentRunner:
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
        run_skip=args.run_skip,
    )
    return IterativeExperimentRunner(
        config_path=experiment_config.source_path,
        project_paths=project_paths,
        training_config=training_config,
    )


def _resolve_runner_cli_options(
    args: argparse.Namespace,
) -> tuple[bool, float, float | None, str, int, int, float, float, int, int, bool]:
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

    device_policy = config_device_policy if getattr(args, "device_policy", None) is None else str(getattr(args, "device_policy"))
    gpu_retries = config_gpu_retries if getattr(args, "gpu_retries", None) is None else int(getattr(args, "gpu_retries"))
    cpu_retries = config_cpu_retries if getattr(args, "cpu_retries", None) is None else int(getattr(args, "cpu_retries"))
    cooldown_after_oom_seconds = config_cooldown_after_oom if getattr(args, "cooldown_after_oom_seconds", None) is None else float(getattr(args, "cooldown_after_oom_seconds"))
    gpu_recovery_cooldown_seconds = config_gpu_recovery_cooldown if getattr(args, "gpu_recovery_cooldown_seconds", None) is None else float(getattr(args, "gpu_recovery_cooldown_seconds"))
    max_consecutive_oom = config_max_consecutive_oom if getattr(args, "max_consecutive_oom", None) is None else int(getattr(args, "max_consecutive_oom"))
    max_task_attempts = config_max_task_attempts if getattr(args, "max_task_attempts", None) is None else int(getattr(args, "max_task_attempts"))
    fail_fast_on_oom = bool(getattr(args, "fail_fast_on_oom", False) or config_fail_fast_on_oom)
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
    _add_optional("--task-cooldown-seconds", task_cooldown_seconds)
    _add_optional("--max-queue-tasks", args.max_queue_tasks)
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


def _print_status_snapshot(snapshot: dict[str, object]) -> None:
    counts = snapshot["counts"]
    current_task = snapshot["current_task"]
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
    if current_task:
        print("Current task: " f"{current_task['preproc_id']} [{current_task['model_name']} - " f"{current_task['param_display']}]")
    if snapshot.get("device_policy") is not None:
        print(f"Device policy: {snapshot['device_policy']}")
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
    print(f"State file: {snapshot['state_path']}")
    print(f"Summary file: {snapshot['summary_path']}")
    print(f"History dir: {snapshot['history_dir']}")
    print(f"Predictions dir: {snapshot['predictions_dir']}")
    print(f"Runner log: {snapshot['log_path']}")
    state_path = Path(str(snapshot["state_path"]))
    if snapshot["total"] == 0 and not state_path.exists():
        print("Note: no persisted runner state exists yet.")


def main(argv: list[str] | None = None) -> int:
    resolved_argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(resolved_argv)

    if args.command == "launch":
        project_paths = _load_project_paths(args)
        store = ExperimentStateStore(project_paths)
        if store.has_active_run():
            snapshot = store.summarize()
            pid = snapshot.get("active_pid")
            print(f"Runner already active with pid={pid}.")
            return 1
        process = launch_background_runner(
            script_path=Path(__file__).resolve(),
            forwarded_args=resolved_argv[1:],
            cwd=PROJECT_ROOT,
            logs_dir=project_paths.experiment_logs_dir,
        )
        command = [sys.executable, str(Path(__file__).resolve()), "run", *resolved_argv[1:]]
        store.write_pid_record(
            config_path=Path(load_experiment_config(args.config).source_path),
            command=command,
            pid=process.process.pid,
            stdout_log_path=process.stdout_path,
            stderr_log_path=process.stderr_path,
        )
        print(f"Background runner started with pid={process.process.pid}.")
        print(f"stdout: {process.stdout_path}")
        print(f"stderr: {process.stderr_path}")
        print("Check status with: python run_experiments.py status")
        return 0

    if args.command == "run":
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

    project_paths = _load_project_paths(args)
    store = ExperimentStateStore(project_paths)

    if args.command == "stop":
        stop_path = store.request_stop()
        print("Stop requested. The runner will finish the current experiment and stop.")
        print(f"Stop flag: {stop_path}")
        return 0

    if args.command == "status":
        snapshot = store.summarize()
        if args.json:
            print(json.dumps(snapshot, indent=2, sort_keys=True))
        else:
            _print_status_snapshot(snapshot)
        return 0

    if args.command == "reset":
        store.reset(purge_results=args.purge_results)
        if args.purge_results:
            print("Runner state and training results were removed.")
        else:
            print("Runner state was reset. Existing training artifacts were preserved.")
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
