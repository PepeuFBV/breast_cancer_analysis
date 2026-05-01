from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from pipeline.experiments import (
    ExperimentStateStore,
    IterativeExperimentRunner,
    IterativeRunOptions,
)
from pipeline.train.preprocessing import PreprocessingTask
from pipeline.train.runner import TrainingConfig, TrainingRunResult
from pipeline.utils.paths import build_project_paths
from scripts import validate_long_runner


def _write_split_csvs(root: Path) -> tuple[Path, Path]:
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    pd.DataFrame(
        {
            "image_path": ["train-0.png", "train-1.png", "train-2.png", "train-3.png"],
            "label": ["1", "2", "1", "2"],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {
            "image_path": ["test-0.png", "test-1.png"],
            "label": ["1", "2"],
        }
    ).to_csv(test_path, index=False)
    return train_path, test_path


def _build_training_config(root: Path) -> tuple[TrainingConfig, object]:
    project_paths = build_project_paths(root / "raw-data", root / "artifacts")
    project_paths.ensure_artifact_dirs()
    train_path, test_path = _write_split_csvs(root)
    config = TrainingConfig(
        train_split_path=train_path,
        test_split_path=test_path,
        history_dir=project_paths.history_dir,
        predictions_dir=project_paths.predictions_dir,
        folds=0,
        validation_size=0.5,
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        loss="categorical_crossentropy",
        model_names=["custom cnn"],
        include_combinations=False,
        run_skip=False,
    )
    return config, project_paths


def _task(index: int) -> PreprocessingTask:
    params = {"index": index}
    return PreprocessingTask(
        preproc_id=f"identity_{index:02d}",
        params=params,
        param_display=f"index={index}",
        param_id=f"index-{index}",
        param_json=json.dumps(params, separators=(",", ":"), sort_keys=True),
        is_combined=False,
        apply=lambda image: image,
    )


def _fake_result(task: PreprocessingTask) -> TrainingRunResult:
    return TrainingRunResult(
        preproc_id=task.preproc_id,
        model_name="custom cnn",
        param_id=task.param_id,
        param_combo=task.param_display,
        param_json=task.param_json,
        best_val_acc=0.9,
        best_epoch=1,
        fold=None,
        history_dict={"accuracy": [0.8], "val_accuracy": [0.9]},
        predictions_df=pd.DataFrame(
            {
                "y_true": [0, 1],
                "y_pred": [0, 1],
                "y_pred_probability": [[0.9, 0.1], [0.2, 0.8]],
                "prob_class_0": [0.9, 0.2],
                "prob_class_1": [0.1, 0.8],
            }
        ),
        selection_strategy="holdout_validation",
        train_samples=4,
        validation_samples=2,
        test_samples=2,
    )


def _build_runner(tmp_path: Path, task_count: int) -> tuple[IterativeExperimentRunner, object]:
    config, project_paths = _build_training_config(tmp_path)
    runner = IterativeExperimentRunner(
        config_path=Path("configs/experiment.smoke.json"),
        project_paths=project_paths,
        training_config=config,
        preprocessing_tasks=[_task(index) for index in range(task_count)],
    )
    return runner, project_paths


@pytest.mark.stress
def test_runner_completes_twenty_mocked_tasks_and_writes_summary(tmp_path) -> None:
    runner, project_paths = _build_runner(tmp_path, task_count=20)

    def fake_run_training_task(task, *args, **kwargs):
        return _fake_result(task.preprocessing_task)

    with patch(
        "pipeline.experiments.runner.run_training_task",
        side_effect=fake_run_training_task,
    ):
        snapshot = runner.run()

    assert snapshot["counts"]["completed"] == 20
    assert snapshot["counts"]["failed"] == 0

    summary_df = pd.read_csv(project_paths.experiment_summary_dir / "experiment_runs.csv")
    assert len(summary_df) == 20
    assert set(summary_df["status"]) == {"completed"}


@pytest.mark.stress
def test_runner_continues_after_ninth_failure(tmp_path) -> None:
    runner, project_paths = _build_runner(tmp_path, task_count=12)
    calls = {"count": 0}

    def fake_run_training_task(task, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 9:
            raise RuntimeError("synthetic ninth failure")
        return _fake_result(task.preprocessing_task)

    with patch(
        "pipeline.experiments.runner.run_training_task",
        side_effect=fake_run_training_task,
    ):
        snapshot = runner.run()

    store = ExperimentStateStore(project_paths)
    state = store.load_state()
    assert snapshot["overall_status"] == "completed_with_failures"
    assert calls["count"] == 12
    assert state["tasks"][8]["status"] == "failed"
    assert [task["status"] for task in state["tasks"][9:]] == [
        "completed",
        "completed",
        "completed",
    ]


@pytest.mark.stress
def test_runner_resume_skips_completed_and_finishes_pending(tmp_path) -> None:
    runner, project_paths = _build_runner(tmp_path, task_count=6)
    calls: list[str] = []

    def fake_run_training_task(task, *args, **kwargs):
        calls.append(task.preproc_id)
        return _fake_result(task.preprocessing_task)

    with patch(
        "pipeline.experiments.runner.run_training_task",
        side_effect=fake_run_training_task,
    ):
        first_snapshot = runner.run(IterativeRunOptions(limit=3, rerun_failed=False, rerun_completed=False))
        second_snapshot = runner.run()

    store = ExperimentStateStore(project_paths)
    state = store.load_state()
    assert first_snapshot["counts"]["completed"] == 3
    assert first_snapshot["counts"]["pending"] == 3
    assert second_snapshot["counts"]["completed"] == 6
    assert calls == [
        "identity_00",
        "identity_01",
        "identity_02",
        "identity_03",
        "identity_04",
        "identity_05",
    ]
    assert [task["status"] for task in state["tasks"]] == ["completed"] * 6


def test_validate_long_runner_main_reports_summary_paths(tmp_path, monkeypatch, capsys) -> None:
    summary_path = tmp_path / "summary.csv"
    state_path = tmp_path / "runner_state.json"
    monkeypatch.setattr(
        validate_long_runner,
        "check_runtime_device",
        lambda **kwargs: SimpleNamespace(
            ok=True,
            requested_device="cpu",
            selected_device="cpu",
            errors=(),
        ),
    )
    monkeypatch.setattr(
        validate_long_runner,
        "_run_validation",
        lambda **kwargs: {
            "snapshot": {
                "overall_status": "completed",
                "total": 20,
                "counts": {
                    "pending": 0,
                    "running": 0,
                    "completed": 20,
                    "failed": 0,
                    "stopped": 0,
                },
                "state_path": str(state_path),
                "summary_path": str(summary_path),
            },
            "summary_path": summary_path,
            "state_path": state_path,
            "peak_process_memory_mb": 128.0,
        },
    )

    exit_code = validate_long_runner.main(
        [
            "--combinations",
            "20",
            "--device",
            "cpu",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Total combinations: 20" in output
    assert str(summary_path) in output
    assert str(state_path) in output
