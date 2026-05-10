#!/usr/bin/env python3
"""Check if the system is ready for unattended experiment execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "artifacts" / "processed"


def _venv_python() -> Path:
    if os.name == "nt":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def _venv_command_hint(script: str) -> str:
    if os.name == "nt":
        return f".\\.venv\\Scripts\\python.exe {script}"
    return f"./.venv/bin/python {script}"


def check_venv() -> tuple[bool, str]:
    """Check if virtualenv exists and is functional."""
    venv_python = _venv_python()
    if not venv_python.exists():
        return False, f"Virtualenv not found. Run: {sys.executable} scripts/bootstrap_env.py"

    result = subprocess.run(
        [str(venv_python), "-c", "import tensorflow"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        return False, f"TensorFlow not installed. Run: {sys.executable} scripts/bootstrap_env.py"

    return True, "Virtualenv is ready"


def check_dataset() -> tuple[bool, str]:
    """Check if dataset is present and valid."""
    result = subprocess.run(
        [str(_venv_python()), "scripts/validate_dataset.py"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        return False, "Dataset validation failed. Check data/INbreast Release 1.0/"

    return True, "Dataset is valid"


def check_preprocessing() -> tuple[bool, str]:
    """Check if data has been preprocessed."""
    train_dir = PROCESSED_DIR / "images" / "train"
    test_dir = PROCESSED_DIR / "images" / "test"

    if not train_dir.exists() or not test_dir.exists():
        return False, f"Data not preprocessed. Run: {_venv_command_hint('preprocess.py')}"

    train_images = list(train_dir.glob("*.png"))
    test_images = list(test_dir.glob("*.png"))

    if not train_images or not test_images:
        return False, f"Preprocessed images not found. Run: {_venv_command_hint('preprocess.py')}"

    return True, f"Data preprocessed ({len(train_images)} train, {len(test_images)} test images)"


def check_gpu() -> tuple[bool, str]:
    """Check GPU availability (non-blocking)."""
    result = subprocess.run(
        [str(_venv_python()), "scripts/check_gpu.py", "--json"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        return False, "GPU check failed (will use CPU)"

    try:
        data = json.loads(result.stdout.decode())
        if data.get("physical_gpus"):
            gpu_count = len(data["physical_gpus"])
            return True, f"GPU available ({gpu_count} device(s))"
        else:
            return False, "No GPU detected (will use CPU)"
    except (json.JSONDecodeError, KeyError):
        return False, "GPU status unknown (will use CPU)"


def main() -> int:
    print("=" * 70)
    print("READINESS CHECK FOR UNATTENDED EXECUTION")
    print("=" * 70)

    checks = [
        ("Virtualenv", check_venv()),
        ("Dataset", check_dataset()),
        ("Preprocessing", check_preprocessing()),
        ("GPU", check_gpu()),
    ]

    all_critical_ok = True

    for name, (ok, message) in checks:
        status = "✓" if ok else "✗"
        print(f"{status} {name:20} {message}")

        # GPU is optional, others are critical
        if not ok and name != "GPU":
            all_critical_ok = False

    print("=" * 70)

    if all_critical_ok:
        print("\n✓ System is ready for unattended execution!")
        print("\nTo start the experiment queue:")
        print(f"  {_venv_command_hint('run_experiments.py launch')}")
        print("\nTo monitor progress:")
        print(f"  {_venv_command_hint('run_experiments.py status')}")
        if os.name == "nt":
            print("  .\\.venv\\Scripts\\streamlit.exe run experiment_dashboard.py")
        else:
            print("  ./.venv/bin/streamlit run experiment_dashboard.py")
        return 0
    else:
        print("\n✗ System is NOT ready. Please fix the issues above.")
        print("\nQuick fix:")
        print(f"  {sys.executable} scripts/unattended_setup.py --gpu auto")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
