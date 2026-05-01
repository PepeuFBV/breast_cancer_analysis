from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pipeline.config import load_experiment_config
from pipeline.evaluate.reporting import generate_final_report
from pipeline.experiments import IterativeExperimentRunner, IterativeRunOptions
from pipeline.train.preprocessing import PreprocessingTask


class _SmokeHistory:
    history = {
        "accuracy": [0.72, 0.82],
        "val_accuracy": [0.7, 0.8],
        "loss": [0.9, 0.5],
        "val_loss": [1.0, 0.6],
    }


class _SmokeModel:
    def fit(self, *args: Any, **kwargs: Any) -> _SmokeHistory:
        return _SmokeHistory()

    def predict(self, model_inputs: np.ndarray, batch_size: int = 8, verbose: int = 0) -> np.ndarray:
        probabilities = np.full((len(model_inputs), 8), 0.01, dtype="float32")
        probabilities[:, 0] = 0.78
        probabilities[:, 1] = 0.15
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        return probabilities


def _smoke_model_builder(*args: Any, **kwargs: Any) -> _SmokeModel:
    return _SmokeModel()


def _write_synthetic_splits(root: Path) -> tuple[Path, Path]:
    image_dir = root / "synthetic-images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    labels = ["1", "1", "2", "2", "1", "2", "1", "2"]
    for index, label in enumerate(labels):
        image_path = image_dir / f"smoke_{index}.png"
        image = np.full((16, 16), index * 20, dtype="uint8")
        cv2.imwrite(str(image_path), image)
        image_paths.append(str(image_path))

    train_df = pd.DataFrame(
        {
            "image_path": image_paths[:6],
            "label": labels[:6],
            "source_id": [f"train-{index}" for index in range(6)],
            "split_group_id": [f"train-{index}" for index in range(6)],
        }
    )
    test_df = pd.DataFrame(
        {
            "image_path": image_paths[6:],
            "label": labels[6:],
            "source_id": [f"test-{index}" for index in range(2)],
            "split_group_id": [f"test-{index}" for index in range(2)],
        }
    )
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train_split.csv"
    test_path = split_dir / "test_split.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def _identity_task(index: int) -> PreprocessingTask:
    preproc_id = f"smoke_identity_{index:02d}"
    return PreprocessingTask(
        preproc_id=preproc_id,
        params={"index": index},
        param_display=f"index={index}",
        param_id=f"index-{index}",
        param_json=f'{{"index":{index}}}',
        is_combined=False,
        apply=lambda image: image,
    )


def _run_smoke(
    *,
    config_path: str | None,
    artifacts_dir: Path,
    work_dir: Path,
    task_count: int,
) -> tuple[dict[str, Any], Path, int]:
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
        validation_size=0.5,
        batch_size=2,
        epochs=2,
        model_names=["custom cnn"],
        preprocessing_ids=["none"],
        include_combinations=False,
        run_skip=False,
    )
    tasks = [_identity_task(index) for index in range(task_count)]
    runner = IterativeExperimentRunner(
        config_path=experiment_config.source_path,
        project_paths=project_paths,
        training_config=training_config,
        model_builders={"custom cnn": _smoke_model_builder},
        preprocessing_tasks=tasks,
    )
    snapshot = runner.run(IterativeRunOptions(limit=task_count))
    evaluation_config = experiment_config.build_evaluation_config(
        project_paths,
        history_dir=project_paths.history_dir,
        predictions_dir=project_paths.predictions_dir,
        output_path=project_paths.final_report_path,
        details_dir=project_paths.reports_dir / "smoke-details",
    )
    final_results, report_path = generate_final_report(evaluation_config)
    return snapshot, report_path, len(final_results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fast synthetic smoke check through runner and evaluation.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Persist smoke artifacts here. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=3,
        help="Number of synthetic runner tasks to execute.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.task_count < 1:
        raise SystemExit("--task-count must be at least 1.")

    if args.artifacts_dir:
        work_dir = Path(args.artifacts_dir).expanduser().resolve() / "_smoke_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
        snapshot, report_path, records = _run_smoke(
            config_path=args.config,
            artifacts_dir=artifacts_dir,
            work_dir=work_dir,
            task_count=args.task_count,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="bca-smoke-") as tmp_dir:
            work_dir = Path(tmp_dir)
            snapshot, report_path, records = _run_smoke(
                config_path=args.config,
                artifacts_dir=work_dir / "artifacts",
                work_dir=work_dir,
                task_count=args.task_count,
            )

    counts = snapshot["counts"]
    print("Smoke run completed.")
    print(f"Completed tasks: {counts['completed']}")
    print(f"Failed tasks: {counts['failed']}")
    print(f"Final report: {report_path}")
    print(f"Evaluation records: {records}")
    return 0 if counts["failed"] == 0 and records == args.task_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
