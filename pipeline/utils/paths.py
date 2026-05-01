from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    raw_data_dir: Path
    artifacts_dir: Path
    processed_dir: Path
    processed_images_dir: Path
    processed_splits_dir: Path
    train_split_path: Path
    test_split_path: Path
    runs_dir: Path
    history_dir: Path
    predictions_dir: Path
    reports_dir: Path
    final_report_path: Path
    loop_log_path: Path
    experiments_dir: Path
    experiment_state_dir: Path
    experiment_logs_dir: Path
    experiment_summary_dir: Path
    experiment_control_dir: Path
    experiment_task_dir: Path

    def ensure_artifact_dirs(self) -> "ProjectPaths":
        for directory in (
            self.artifacts_dir,
            self.processed_dir,
            self.processed_images_dir,
            self.processed_splits_dir,
            self.runs_dir,
            self.history_dir,
            self.predictions_dir,
            self.reports_dir,
            self.loop_log_path.parent,
            self.experiments_dir,
            self.experiment_state_dir,
            self.experiment_logs_dir,
            self.experiment_summary_dir,
            self.experiment_control_dir,
            self.experiment_task_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def build_project_paths(
    raw_data_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> ProjectPaths:
    raw_data_root = Path(raw_data_dir) if raw_data_dir else PROJECT_ROOT / "data" / "INbreast Release 1.0"
    artifacts_root = Path(artifacts_dir) if artifacts_dir else PROJECT_ROOT / "artifacts"

    processed_dir = artifacts_root / "processed"
    processed_images_dir = processed_dir / "images"
    processed_splits_dir = processed_dir / "splits"
    runs_dir = artifacts_root / "runs"
    reports_dir = artifacts_root / "reports"
    experiments_dir = artifacts_root / "experiments"

    return ProjectPaths(
        project_root=PROJECT_ROOT,
        raw_data_dir=raw_data_root,
        artifacts_dir=artifacts_root,
        processed_dir=processed_dir,
        processed_images_dir=processed_images_dir,
        processed_splits_dir=processed_splits_dir,
        train_split_path=processed_splits_dir / "train_split.csv",
        test_split_path=processed_splits_dir / "test_split.csv",
        runs_dir=runs_dir,
        history_dir=runs_dir / "history",
        predictions_dir=runs_dir / "predictions",
        reports_dir=reports_dir,
        final_report_path=reports_dir / "final_comprehensive_results.csv",
        loop_log_path=runs_dir / "logs" / "train-loop.log",
        experiments_dir=experiments_dir,
        experiment_state_dir=experiments_dir / "state",
        experiment_logs_dir=experiments_dir / "logs",
        experiment_summary_dir=experiments_dir / "summary",
        experiment_control_dir=experiments_dir / "control",
        experiment_task_dir=experiments_dir / "tasks",
    )
