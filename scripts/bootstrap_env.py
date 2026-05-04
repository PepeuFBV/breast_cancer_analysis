from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV_DIR = PROJECT_ROOT / ".venv"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "INbreast Release 1.0"
WSL_DRIVER_DIR = Path("/usr/lib/wsl/lib")
WSL_NVIDIA_SMI = WSL_DRIVER_DIR / "nvidia-smi"


class BootstrapError(RuntimeError):
    """Raised when the environment cannot be bootstrapped as requested."""


@dataclass(frozen=True)
class BootstrapPlan:
    tensorflow_requirement: str
    require_gpu_check: bool
    gpu_driver_visible: bool


def _command_environment() -> dict[str, str]:
    environment = dict(os.environ)
    if WSL_DRIVER_DIR.exists():
        path_entries = [entry for entry in environment.get("PATH", "").split(":") if entry]
        driver_dir = str(WSL_DRIVER_DIR)
        if driver_dir not in path_entries:
            environment["PATH"] = ":".join([driver_dir, *path_entries])
    return environment


def _run(command: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    subprocess.run(command, check=True, cwd=cwd, env=_command_environment())


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _ensure_virtualenv(venv_dir: Path, python_executable: str) -> Path:
    venv_python = _venv_python(venv_dir)
    if venv_python.exists():
        return venv_python

    _run([python_executable, "-m", "venv", str(venv_dir)])
    if not venv_python.exists():
        raise BootstrapError(f"Virtualenv creation succeeded but {venv_python} is missing.")
    return venv_python


def _nvidia_smi_command() -> list[str] | None:
    if WSL_NVIDIA_SMI.exists():
        return [str(WSL_NVIDIA_SMI)]

    executable = shutil.which("nvidia-smi")
    if executable:
        return [executable]
    return None


def gpu_driver_visible() -> bool:
    command = _nvidia_smi_command()
    if command is None:
        return False

    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_command_environment(),
    )
    return result.returncode == 0


def resolve_bootstrap_plan(
    gpu_mode: str,
    *,
    platform_name: str | None = None,
    driver_visible: bool | None = None,
) -> BootstrapPlan:
    platform_name = platform_name or sys.platform
    driver_visible = gpu_driver_visible() if driver_visible is None else driver_visible

    if gpu_mode == "off":
        return BootstrapPlan(
            tensorflow_requirement="tensorflow",
            require_gpu_check=False,
            gpu_driver_visible=driver_visible,
        )

    if platform_name != "linux":
        if gpu_mode == "required":
            raise BootstrapError(
                "TensorFlow GPU bootstrap in this script is only supported on Linux/WSL2. "
                "For native Windows TensorFlow 2.10 GPU, use `powershell -File scripts/bootstrap_windows_gpu.ps1`."
            )
        return BootstrapPlan(
            tensorflow_requirement="tensorflow",
            require_gpu_check=False,
            gpu_driver_visible=driver_visible,
        )

    if driver_visible:
        return BootstrapPlan(
            tensorflow_requirement="tensorflow[and-cuda]",
            require_gpu_check=True,
            gpu_driver_visible=True,
        )

    if gpu_mode == "required":
        raise BootstrapError("No NVIDIA GPU driver is visible from this Linux environment. " "On WSL2, confirm `/usr/lib/wsl/lib/nvidia-smi` works first.")

    return BootstrapPlan(
        tensorflow_requirement="tensorflow",
        require_gpu_check=False,
        gpu_driver_visible=False,
    )


def _requirement_name(requirement: str) -> str | None:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if not match:
        return None
    return match.group(1).lower().replace("_", "-")


def render_requirements(requirements_text: str, tensorflow_requirement: str) -> str:
    rendered_lines: list[str] = []
    replaced_tensorflow = False

    for line in requirements_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            rendered_lines.append(line)
            continue

        requirement = stripped.split("#", 1)[0].strip()
        name = _requirement_name(requirement)
        if name in {"tensorflow", "tensorflow-cpu"}:
            if not replaced_tensorflow:
                rendered_lines.append(tensorflow_requirement)
                replaced_tensorflow = True
            continue

        rendered_lines.append(line)

    if not replaced_tensorflow:
        rendered_lines.append(tensorflow_requirement)

    return "\n".join(rendered_lines) + "\n"


def install_runtime(
    *,
    venv_python: Path,
    tensorflow_requirement: str,
    install_dev: bool,
) -> None:
    requirements_path = PROJECT_ROOT / "requirements.txt"
    requirements_text = requirements_path.read_text(encoding="utf-8")
    rendered_requirements = render_requirements(
        requirements_text=requirements_text,
        tensorflow_requirement=tensorflow_requirement,
    )

    _run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])

    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".requirements.txt",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(rendered_requirements)
        temp_requirements = Path(handle.name)

    try:
        _run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "-r",
                str(temp_requirements),
            ]
        )
    finally:
        temp_requirements.unlink(missing_ok=True)

    _run([str(venv_python), "-m", "pip", "install", "-e", str(PROJECT_ROOT)])

    if install_dev:
        _run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "-r",
                str(PROJECT_ROOT / "requirements-dev.txt"),
            ]
        )


def _dataset_check_enabled(dataset_dir: Path, skip_dataset: bool) -> bool:
    return not skip_dataset and dataset_dir.exists()


def run_validation_checks(
    *,
    venv_python: Path,
    require_gpu: bool,
    raw_data_dir: Path,
    artifacts_dir: Path | None,
    skip_dataset: bool,
) -> bool:
    """Run validation checks and return True if all passed, False otherwise."""
    environment_check = [
        str(venv_python),
        str(PROJECT_ROOT / "scripts" / "check_environment.py"),
        "--require-venv",
    ]
    if artifacts_dir is not None:
        environment_check.extend(["--artifacts-dir", str(artifacts_dir)])
    if raw_data_dir != DEFAULT_DATASET_DIR:
        environment_check.extend(["--raw-data-dir", str(raw_data_dir)])
    if not _dataset_check_enabled(raw_data_dir, skip_dataset):
        environment_check.append("--skip-dataset")
    _run(environment_check)

    gpu_check = [str(venv_python), str(PROJECT_ROOT / "scripts" / "check_gpu.py")]
    if require_gpu:
        gpu_check.append("--require-gpu")

    result = subprocess.run(
        gpu_check,
        check=False,
        cwd=PROJECT_ROOT,
        env=_command_environment(),
    )
    return result.returncode == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Create or repair the project virtualenv and install runtime " "dependencies."))
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use when creating the virtualenv.",
    )
    parser.add_argument(
        "--venv-dir",
        default=str(DEFAULT_VENV_DIR),
        help="Virtualenv directory to create or reuse.",
    )
    parser.add_argument(
        "--gpu",
        choices=("auto", "required", "off"),
        default="auto",
        help="Install GPU TensorFlow when possible, require it, or force CPU mode.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development dependencies from requirements-dev.txt.",
    )
    parser.add_argument(
        "--raw-data-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Dataset directory used for the post-install validation check.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Artifacts directory used for the post-install validation check.",
    )
    parser.add_argument(
        "--skip-dataset-check",
        action="store_true",
        help="Skip dataset validation after installing dependencies.",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Install the environment but skip post-install validation checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    venv_dir = Path(args.venv_dir).expanduser().resolve()
    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir is not None else None

    try:
        plan = resolve_bootstrap_plan(args.gpu)
        venv_python = _ensure_virtualenv(venv_dir, args.python)
        install_runtime(
            venv_python=venv_python,
            tensorflow_requirement=plan.tensorflow_requirement,
            install_dev=args.dev,
        )

        gpu_check_passed = True
        if not args.skip_checks:
            gpu_check_passed = run_validation_checks(
                venv_python=venv_python,
                require_gpu=plan.require_gpu_check,
                raw_data_dir=raw_data_dir,
                artifacts_dir=artifacts_dir,
                skip_dataset=args.skip_dataset_check,
            )
    except BootstrapError as error:
        print(f"Bootstrap failed: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(
            f"Bootstrap command failed with exit code {error.returncode}: " f"{' '.join(error.cmd)}",
            file=sys.stderr,
        )
        return error.returncode or 1

    print(f"Bootstrap complete. Virtualenv: {venv_dir}")
    print(f"TensorFlow requirement: {plan.tensorflow_requirement}")
    if plan.require_gpu_check:
        if gpu_check_passed:
            print("GPU validation: required and passed")
        else:
            print("GPU validation: required but failed")
            return 1
    else:
        if gpu_check_passed:
            print("GPU validation: optional and passed")
        else:
            print("GPU validation: optional (GPU not available, will use CPU)")
            if plan.tensorflow_requirement == "tensorflow[and-cuda]":
                print("Note: tensorflow[and-cuda] is installed but GPU is not accessible.\n" "      The pipeline will run on CPU. For GPU support, see docs/wsl_gpu_setup.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
