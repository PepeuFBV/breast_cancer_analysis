from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pipeline.config import load_experiment_config
from pipeline.experiments import IterativeExperimentRunner, IterativeRunOptions
from pipeline.train.preprocessing import PreprocessingTask
from pipeline.utils.memory import get_process_memory_mb
from pipeline.utils.runtime_device import check_runtime_device

DEFAULT_CONFIG_PATH = "configs/experiment.smoke.json"

def _write_synthetic_splits(root: Path) -> tuple[Path, Path]:
    image_dir = root / "synthetic-images"
    image_dir.mkdir(parents=True, exist_ok=True)

    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    labels = ["1", "2"] * 8
    for index, label in enumerate(labels):
        image_path = image_dir / f"sample_{index:02d}.png"
        image = np.zeros((16, 16), dtype="uint8")
        image[:, :] = 20 + (index * 10)
        if label == "2":
            image[4:12, 4:12] = min(255, 80 + index * 8)
        cv2.imwrite(str(image_path), image)
        record = {
            "image_path": str(image_path),
            "label": label,
            "source_id": f"synthetic-{index:02d}",
            "split_group_id": f"synthetic-{index:02d}",
        }
        if index < 12:
            train_records.append(record)
        else:
            test_records.append(record)

    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train_split.csv"
    test_path = split_dir / "test_split.csv"
    pd.DataFrame(train_records).to_csv(train_path, index=False)
    pd.DataFrame(test_records).to_csv(test_path, index=False)
    return train_path, test_path


def _identity_task(index: int) -> PreprocessingTask:
    return PreprocessingTask(
        preproc_id=f"long_runner_identity_{index:02d}",
        params={"index": index},
        param_display=f"index={index}",
        param_id=f"index-{index}",
        param_json=f'{{"index":{index}}}',
        is_combined=False,
        apply=lambda image: image,
    )


def _skipped_count(snapshot: dict[str, Any]) -> int:
    counts = snapshot["counts"]
    tracked = sum(counts.get(status, 0) for status in counts)
    return int(snapshot["total"]) - tracked


def _run_validation(
    *,
    config_path: str,
    artifacts_dir: Path,
    work_dir: Path,
    combinations: int,
) -> dict[str, Any]:
    experiment_config = load_experiment_config(config_path)
    project_paths = experiment_config.resolve_project_paths(
        raw_data_dir=work_dir / "raw-data",
        artifacts_dir=artifacts_dir,
    ).ensure_artifact_dirs()
    train_split, test_split = _write_synthetic_splits(work_dir)
    training_config = experiment_config.build_training_config(
        project_paths,
        train_split=train_split,
        test_split=test_split,
        folds=0,
        validation_size=0.25,
        batch_size=2,
        epochs=1,
        model_names=["custom cnn"],
        preprocessing_ids=["none"],
        include_combinations=False,
        run_skip=False,
    )
    tasks = [_identity_task(index) for index in range(combinations)]
    runner = IterativeExperimentRunner(
        config_path=experiment_config.source_path,
        project_paths=project_paths,
        training_config=training_config,
        preprocessing_tasks=tasks,
    )
    snapshot = runner.run(IterativeRunOptions(limit=combinations))
    summary_path = Path(str(snapshot["summary_path"]))
    state_path = Path(str(snapshot["state_path"]))
    peak_memory = snapshot.get("peak_process_memory_mb")
    if peak_memory is None:
        peak_memory = get_process_memory_mb()
    return {
        "snapshot": snapshot,
        "summary_path": summary_path,
        "state_path": state_path,
        "peak_process_memory_mb": peak_memory,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate long-running experiment orchestration with tiny real "
            "training tasks."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Experiment config used for tiny training defaults.",
    )
    parser.add_argument(
        "--combinations",
        type=int,
        default=20,
        help="How many tiny training tasks to execute.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Runtime device policy for the validation run.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if TensorFlow cannot use GPU for this validation.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Persist validation artifacts here instead of using a temporary directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.combinations < 1:
        raise SystemExit("--combinations must be at least 1.")

    runtime_check = check_runtime_device(
        device=args.device,
        require_gpu=args.require_gpu,
    )
    if not runtime_check.ok:
        print(f"Runtime validation failed for device={args.device}.")
        for error in runtime_check.errors:
            print(f"- {error}")
        return 1

    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
        work_dir = artifacts_dir / "_validate_long_runner"
        work_dir.mkdir(parents=True, exist_ok=True)
        result = _run_validation(
            config_path=args.config,
            artifacts_dir=artifacts_dir,
            work_dir=work_dir,
            combinations=args.combinations,
        )
    else:
        root = Path(tempfile.mkdtemp(prefix="bca-long-runner-"))
        result = _run_validation(
            config_path=args.config,
            artifacts_dir=root / "artifacts",
            work_dir=root / "work",
            combinations=args.combinations,
        )

    snapshot = result["snapshot"]
    counts = snapshot["counts"]
    completed = int(counts["completed"])
    failed = int(counts["failed"])
    total = int(snapshot["total"])
    skipped = _skipped_count(snapshot)
    print("Long runner validation completed.")
    print(f"Requested device: {runtime_check.requested_device}")
    print(f"Selected device: {runtime_check.selected_device}")
    print(f"Overall status: {snapshot['overall_status']}")
    print(f"Total combinations: {total}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Peak memory (MB): {result['peak_process_memory_mb']}")
    print(f"Final state path: {result['state_path']}")
    print(f"Summary CSV path: {result['summary_path']}")
    is_success = failed == 0 and completed >= args.combinations
    if not is_success:
        print(
            "Validation failed: expected all requested combinations to complete "
            f"(requested={args.combinations}, completed={completed}, failed={failed})."
        )
    return 0 if is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
