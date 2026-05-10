from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest

TEST_TASK_SHIM_PATH_ENV = "BREAST_CANCER_ANALYSIS_TEST_TASK_SHIM_PATH"


def _write_splits(root: Path) -> tuple[Path, Path]:
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    labels: list[str] = []
    for index in range(8):
        image_path = image_dir / f"image_{index}.png"
        cv2.imwrite(str(image_path), np.full((16, 16), index * 10, dtype="uint8"))
        image_paths.append(str(image_path))
        labels.append("1" if index < 4 else "2")

    train_path = root / "train.csv"
    test_path = root / "test.csv"
    pd.DataFrame({"image_path": image_paths[:6], "label": labels[:6]}).to_csv(train_path, index=False)
    pd.DataFrame({"image_path": image_paths[6:], "label": labels[6:]}).to_csv(test_path, index=False)
    return train_path, test_path


def _run_cli(*args: str, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "run_experiments.py", *args]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError("Command failed.\n" f"command={' '.join(command)}\n" f"returncode={completed.returncode}\n" f"stdout={completed.stdout}\n" f"stderr={completed.stderr}")
    return completed


def _status_json(*, env: dict[str, str], artifacts_dir: Path) -> dict[str, Any]:
    completed = _run_cli(
        "status",
        "--json",
        "--artifacts-dir",
        str(artifacts_dir),
        env=env,
    )
    return json.loads(completed.stdout)


def _partial_json(*, env: dict[str, str], artifacts_dir: Path, max_rows: int = 500) -> dict[str, Any]:
    completed = _run_cli(
        "partial",
        "--artifacts-dir",
        str(artifacts_dir),
        "--max-rows",
        str(max_rows),
        env=env,
    )
    return json.loads(completed.stdout)


def _wait_for(
    predicate,
    *,
    timeout_seconds: float = 30.0,
    interval_seconds: float = 0.2,
) -> Any:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval_seconds)
    raise AssertionError("Timed out waiting for condition.")


def _launch_args(
    *,
    train_split: Path,
    test_split: Path,
    artifacts_dir: Path,
    limit: int,
    config_path: str = "configs/experiment.default.json",
    preprocessing_ids: list[str] | None = None,
    augmentation_values: list[int] | None = None,
) -> list[str]:
    resolved_preprocessing_ids = preprocessing_ids or ["none", "denoise"]
    resolved_augmentation_values = augmentation_values or [1]
    return [
        "launch",
        "--config",
        config_path,
        "--artifacts-dir",
        str(artifacts_dir),
        "--raw-data-dir",
        str(artifacts_dir / "raw-data"),
        "--train-split",
        str(train_split),
        "--test-split",
        str(test_split),
        "--models",
        "custom cnn",
        "--preprocessing",
        *resolved_preprocessing_ids,
        "--no-combined-preprocessing",
        "--augmentations-per-image",
        *[str(value) for value in resolved_augmentation_values],
        "--folds",
        "0",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--validation-size",
        "0.5",
        "--learning-rate",
        "0.0001",
        "--loss",
        "categorical_crossentropy",
        "--no-run-skip",
        "--isolate-tasks",
        "--task-cooldown-seconds",
        "0",
        "--device-policy",
        "cpu-only",
        "--gpu-retries",
        "0",
        "--cpu-retries",
        "0",
        "--max-task-attempts",
        "1",
        "--limit",
        str(limit),
    ]


@pytest.mark.integration
def test_launch_stop_resume_with_partial_consistency(tmp_path: Path) -> None:
    train_split, test_split = _write_splits(tmp_path / "data")
    artifacts_dir = tmp_path / "artifacts"
    shim_path = tmp_path / "task-shim.json"
    shim_path.write_text(
        json.dumps(
            {
                "default": {"status": "completed", "sleep_seconds": 2.0, "result_summary": {"best_val_acc": 0.9}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env[TEST_TASK_SHIM_PATH_ENV] = str(shim_path)

    _run_cli(
        *_launch_args(
            train_split=train_split,
            test_split=test_split,
            artifacts_dir=artifacts_dir,
            limit=3,
            config_path="configs/experiment.smoke.json",
            preprocessing_ids=["none"],
            augmentation_values=[1, 2, 3],
        ),
        env=env,
    )

    _wait_for(
        lambda: (
            snapshot
            if (
                (snapshot := _status_json(env=env, artifacts_dir=artifacts_dir)).get("counts", {}).get("running", 0) >= 1
                and snapshot.get("active_pid") is not None
            )
            else None
        ),
        timeout_seconds=30.0,
    )

    partial_running = _partial_json(env=env, artifacts_dir=artifacts_dir, max_rows=10)
    assert partial_running["row_filter"] == "finalized_only"
    assert partial_running["rows_returned"] >= 0
    assert "running" in partial_running["counts"]
    assert partial_running["total"] == 3

    _run_cli("stop", "--artifacts-dir", str(artifacts_dir), env=env)
    stopped_snapshot = _wait_for(
        lambda: (snapshot if ((snapshot := _status_json(env=env, artifacts_dir=artifacts_dir)).get("overall_status") in {"paused", "stopped", "idle", "completed_with_failures"} and snapshot.get("active_pid") is None) else None),
        timeout_seconds=30.0,
    )
    assert stopped_snapshot["counts"]["pending"] >= 1

    shim_path.write_text(
        json.dumps(
            {
                "default": {"status": "completed", "sleep_seconds": 0.1, "result_summary": {"best_val_acc": 0.9}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _run_cli(
        *_launch_args(
            train_split=train_split,
            test_split=test_split,
            artifacts_dir=artifacts_dir,
            limit=3,
            config_path="configs/experiment.smoke.json",
            preprocessing_ids=["none"],
            augmentation_values=[1, 2, 3],
        ),
        env=env,
    )
    final_snapshot = _wait_for(
        lambda: (
            snapshot
            if (
                (snapshot := _status_json(env=env, artifacts_dir=artifacts_dir)).get("counts", {}).get("completed", 0) == 3
                and snapshot.get("counts", {}).get("pending", 0) == 0
                and snapshot.get("counts", {}).get("running", 0) == 0
                and snapshot.get("active_pid") is None
            )
            else None
        ),
        timeout_seconds=90.0,
    )
    assert final_snapshot["counts"]["completed"] == 3
    assert final_snapshot["counts"]["pending"] == 0
    assert final_snapshot["desired_state"] == "running"

    partial_final = _partial_json(env=env, artifacts_dir=artifacts_dir, max_rows=10)
    assert partial_final["rows_returned"] == 3
    assert partial_final["counts"]["completed"] == 3
    assert partial_final["counts"]["pending"] == 0


@pytest.mark.integration
def test_stale_pid_recovery_after_external_kill(tmp_path: Path) -> None:
    train_split, test_split = _write_splits(tmp_path / "data")
    artifacts_dir = tmp_path / "artifacts"
    shim_path = tmp_path / "task-shim.json"
    shim_path.write_text(
        json.dumps(
            {
                "default": {"status": "completed", "sleep_seconds": 4.0, "result_summary": {"best_val_acc": 0.75}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env[TEST_TASK_SHIM_PATH_ENV] = str(shim_path)

    _run_cli(
        *_launch_args(
            train_split=train_split,
            test_split=test_split,
            artifacts_dir=artifacts_dir,
            limit=1,
            config_path="configs/experiment.smoke.json",
            preprocessing_ids=["none"],
            augmentation_values=[1],
        ),
        env=env,
    )

    running_snapshot = _wait_for(
        lambda: (snapshot if ((snapshot := _status_json(env=env, artifacts_dir=artifacts_dir)).get("active_pid") is not None and snapshot.get("counts", {}).get("running", 0) == 1) else None),
        timeout_seconds=20.0,
    )
    pid = int(running_snapshot["active_pid"])

    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=False, capture_output=True, text=True)
    else:
        os.kill(pid, signal.SIGKILL)

    post_kill_snapshot = _wait_for(
        lambda: (snapshot if ((snapshot := _status_json(env=env, artifacts_dir=artifacts_dir)).get("active_pid") is None and snapshot.get("counts", {}).get("stopped", 0) >= 1) else None),
        timeout_seconds=30.0,
    )
    assert post_kill_snapshot["overall_status"] in {"stopped", "idle"}

    shim_path.write_text(
        json.dumps(
            {
                "default": {"status": "completed", "sleep_seconds": 0.1, "result_summary": {"best_val_acc": 0.75}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _run_cli(
        *_launch_args(
            train_split=train_split,
            test_split=test_split,
            artifacts_dir=artifacts_dir,
            limit=1,
            config_path="configs/experiment.smoke.json",
            preprocessing_ids=["none"],
            augmentation_values=[1],
        ),
        env=env,
    )
    final_snapshot = _wait_for(
        lambda: (snapshot if ((snapshot := _status_json(env=env, artifacts_dir=artifacts_dir)).get("counts", {}).get("completed", 0) == 1 and snapshot.get("counts", {}).get("running", 0) == 0 and snapshot.get("active_pid") is None) else None),
        timeout_seconds=90.0,
    )
    assert final_snapshot["counts"]["completed"] == 1
    assert final_snapshot["counts"]["stopped"] == 0
    assert final_snapshot["last_recovery_reason"] is not None
    assert final_snapshot["last_recovery_at"] is not None
