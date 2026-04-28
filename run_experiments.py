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
)
from pipeline.utils.gpu_env import ensure_tensorflow_wsl_gpu_env
from train import add_training_runtime_arguments

PROJECT_ROOT = Path(__file__).resolve().parent

ensure_tensorflow_wsl_gpu_env()


def _add_resolution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to an experiment JSON config. "
            "Defaults to configs/experiment.default.json."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Artifacts root directory. Defaults to artifacts/.",
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    add_training_runtime_arguments(parser)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Control iterative, resumable execution of training experiments.")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run or resume experiments.")
    _add_run_arguments(run_parser)

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
    return experiment_config.resolve_project_paths(
        artifacts_dir=args.artifacts_dir
    ).ensure_artifact_dirs()


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
    if current_task:
        print(
            "Current task: "
            f"{current_task['preproc_id']} [{current_task['model_name']} - "
            f"{current_task['param_display']}]"
        )
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
        )
        print(f"Background runner started with pid={process.pid}.")
        return 0

    if args.command == "run":
        runner = _build_runner(args)
        try:
            snapshot = runner.run(
                IterativeRunOptions(
                    rerun_failed=args.rerun_failed,
                    rerun_completed=args.rerun_completed,
                    limit=args.limit,
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
