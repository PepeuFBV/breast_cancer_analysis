#!/usr/bin/env python3
"""Unattended setup for the breast cancer analysis pipeline.

This script prepares the environment, validates the dataset, and ensures
everything is ready for running experiments without user intervention.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(command)}")
    return subprocess.run(command, check=check, cwd=PROJECT_ROOT)


def bootstrap_environment(gpu_mode: str, install_dev: bool) -> bool:
    """Bootstrap the Python environment. Returns True if successful."""
    print("\n" + "=" * 70)
    print("STEP 1: Bootstrapping Python environment")
    print("=" * 70)

    cmd = ["python3", "scripts/bootstrap_env.py", "--gpu", gpu_mode]
    if install_dev:
        cmd.append("--dev")

    result = _run(cmd, check=False)
    return result.returncode == 0


def validate_dataset() -> bool:
    """Validate the dataset. Returns True if successful."""
    print("\n" + "=" * 70)
    print("STEP 2: Validating dataset")
    print("=" * 70)

    result = _run([str(VENV_PYTHON), "scripts/validate_dataset.py"], check=False)
    return result.returncode == 0


def run_smoke_tests() -> bool:
    """Run smoke tests. Returns True if successful."""
    print("\n" + "=" * 70)
    print("STEP 3: Running smoke tests")
    print("=" * 70)

    result = _run([str(VENV_PYTHON), "scripts/smoke_run.py"], check=False)
    return result.returncode == 0


def preprocess_data() -> bool:
    """Preprocess the dataset. Returns True if successful."""
    print("\n" + "=" * 70)
    print("STEP 4: Preprocessing dataset")
    print("=" * 70)

    result = _run([str(VENV_PYTHON), "preprocess.py"], check=False)
    return result.returncode == 0


def print_summary(
    bootstrap_ok: bool,
    dataset_ok: bool,
    smoke_ok: bool,
    preprocess_ok: bool,
    skip_preprocess: bool,
) -> None:
    """Print setup summary."""
    print("\n" + "=" * 70)
    print("SETUP SUMMARY")
    print("=" * 70)
    print(f"Environment bootstrap: {'✓ PASSED' if bootstrap_ok else '✗ FAILED'}")
    print(f"Dataset validation:    {'✓ PASSED' if dataset_ok else '✗ FAILED'}")
    print(f"Smoke tests:           {'✓ PASSED' if smoke_ok else '✗ FAILED'}")
    if not skip_preprocess:
        print(f"Data preprocessing:    {'✓ PASSED' if preprocess_ok else '✗ FAILED'}")
    else:
        print("Data preprocessing:    SKIPPED")
    print("=" * 70)


def print_next_steps(all_ok: bool, skip_preprocess: bool) -> None:
    """Print next steps for the user."""
    if all_ok:
        print("\n✓ Setup complete! Ready for unattended execution.")
        print("\nTo run the full experiment queue:")
        if skip_preprocess:
            print("  ./.venv/bin/python preprocess.py")
        print("  ./.venv/bin/python run_experiments.py launch")
        print("\nTo monitor progress:")
        print("  ./.venv/bin/python run_experiments.py status")
        print("  ./.venv/bin/streamlit run experiment_dashboard.py")
        print("\nTo generate the final report after completion:")
        print("  ./.venv/bin/python evaluate.py")
    else:
        print("\n✗ Setup incomplete. Please fix the errors above.")
        print("\nFor troubleshooting:")
        print("  See docs/troubleshooting.md")
        print("  See docs/wsl_gpu_setup.md (for WSL GPU issues)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unattended setup for the breast cancer analysis pipeline.")
    parser.add_argument(
        "--gpu",
        choices=("auto", "required", "off"),
        default="auto",
        help="GPU mode: auto (use if available), required (fail if unavailable), " "or off (CPU only). Default: auto",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development dependencies (tests, linting, formatting).",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip smoke tests (faster but less validation).",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip data preprocessing (run it manually later).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("=" * 70)
    print("UNATTENDED SETUP FOR BREAST CANCER ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"GPU mode: {args.gpu}")
    print(f"Install dev dependencies: {args.dev}")
    print(f"Skip smoke tests: {args.skip_smoke}")
    print(f"Skip preprocessing: {args.skip_preprocess}")

    # Step 1: Bootstrap environment
    bootstrap_ok = bootstrap_environment(args.gpu, args.dev)
    if not bootstrap_ok:
        print("\n✗ Environment bootstrap failed.", file=sys.stderr)
        print("Cannot proceed with setup.", file=sys.stderr)
        return 1

    # Step 2: Validate dataset
    dataset_ok = validate_dataset()
    if not dataset_ok:
        print("\n✗ Dataset validation failed.", file=sys.stderr)
        print("Ensure the INbreast dataset is in data/INbreast Release 1.0/")
        print_summary(bootstrap_ok, dataset_ok, False, False, args.skip_preprocess)
        print_next_steps(False, args.skip_preprocess)
        return 1

    # Step 3: Run smoke tests (optional)
    smoke_ok = True
    if not args.skip_smoke:
        smoke_ok = run_smoke_tests()
        if not smoke_ok:
            print("\n⚠ Smoke tests failed.", file=sys.stderr)
            print("The environment may still work, but validation failed.")

    # Step 4: Preprocess data (optional)
    preprocess_ok = True
    if not args.skip_preprocess:
        preprocess_ok = preprocess_data()
        if not preprocess_ok:
            print("\n✗ Data preprocessing failed.", file=sys.stderr)

    # Summary
    all_ok = bootstrap_ok and dataset_ok and smoke_ok and preprocess_ok
    print_summary(bootstrap_ok, dataset_ok, smoke_ok, preprocess_ok, args.skip_preprocess)
    print_next_steps(all_ok, args.skip_preprocess)

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
