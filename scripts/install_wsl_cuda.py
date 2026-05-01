#!/usr/bin/env python3
"""Install CUDA toolkit and cuDNN for TensorFlow GPU support in WSL2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _is_wsl_linux() -> bool:
    if sys.platform != "linux":
        return False
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"Running: {' '.join(command)}")
    return subprocess.run(command, check=check)


def check_nvidia_smi() -> bool:
    """Check if nvidia-smi is accessible."""
    wsl_nvidia_smi = Path("/usr/lib/wsl/lib/nvidia-smi")
    if wsl_nvidia_smi.exists():
        result = subprocess.run(
            [str(wsl_nvidia_smi)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    result = subprocess.run(
        ["nvidia-smi"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def install_cuda_toolkit(cuda_version: str = "12-3") -> None:
    """Install CUDA toolkit for WSL."""
    print(f"\nInstalling CUDA toolkit {cuda_version}...")

    # Download and install CUDA keyring
    keyring_url = "https://developer.download.nvidia.com/compute/cuda/repos/" "wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
    keyring_file = "/tmp/cuda-keyring_1.1-1_all.deb"

    _run(["wget", "-O", keyring_file, keyring_url])
    _run(["sudo", "dpkg", "-i", keyring_file])
    _run(["sudo", "apt", "update"])
    _run(["sudo", "apt", "install", "-y", f"cuda-toolkit-{cuda_version}"])

    print(f"CUDA toolkit {cuda_version} installed successfully.")


def install_cudnn() -> None:
    """Install cuDNN libraries."""
    print("\nInstalling cuDNN libraries...")
    _run(["sudo", "apt", "install", "-y", "libcudnn8", "libcudnn8-dev"])
    print("cuDNN libraries installed successfully.")


def verify_installation() -> bool:
    """Verify CUDA installation."""
    print("\nVerifying CUDA installation...")
    result = subprocess.run(
        ["nvcc", "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        print(result.stdout.decode())
        return True
    else:
        print("nvcc not found. CUDA toolkit may not be in PATH.")
        print("Add to ~/.bashrc: export PATH=/usr/local/cuda/bin:$PATH")
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install CUDA toolkit and cuDNN for TensorFlow GPU in WSL2.")
    parser.add_argument(
        "--cuda-version",
        default="12-3",
        help="CUDA version to install (e.g., 12-3 for CUDA 12.3). Default: 12-3",
    )
    parser.add_argument(
        "--skip-cudnn",
        action="store_true",
        help="Skip cuDNN installation (tensorflow[and-cuda] includes it).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing installation without installing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not _is_wsl_linux():
        print("This script is designed for WSL2 on Linux.", file=sys.stderr)
        print("For native Linux, follow your distribution's CUDA installation guide.")
        return 1

    print("WSL2 detected. Checking GPU visibility...")
    if not check_nvidia_smi():
        print(
            "\nERROR: nvidia-smi is not accessible from WSL.",
            file=sys.stderr,
        )
        print("Ensure:", file=sys.stderr)
        print("1. Windows NVIDIA drivers are up to date (510.06+)", file=sys.stderr)
        print("2. WSL2 is updated: wsl --update", file=sys.stderr)
        print("3. /usr/lib/wsl/lib/nvidia-smi exists and works", file=sys.stderr)
        return 1

    print("GPU is visible via nvidia-smi.")

    if args.verify_only:
        return 0 if verify_installation() else 1

    try:
        install_cuda_toolkit(args.cuda_version)
        if not args.skip_cudnn:
            install_cudnn()
        verify_installation()

        print("\n" + "=" * 70)
        print("Installation complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Add CUDA to PATH (if not already):")
        print("   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc")
        print("   source ~/.bashrc")
        print("\n2. Reinstall the project environment:")
        print("   python3 scripts/bootstrap_env.py --gpu auto")
        print("\n3. Verify TensorFlow GPU:")
        print("   ./.venv/bin/python scripts/check_gpu.py --require-gpu")
        print("\nFor more details, see docs/wsl_gpu_setup.md")

        return 0
    except subprocess.CalledProcessError as error:
        print(f"\nInstallation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
