from __future__ import annotations

import os
import sys
from pathlib import Path

_GPU_ENV_BOOTSTRAPPED = "BREAST_CANCER_ANALYSIS_GPU_ENV_BOOTSTRAPPED"


def _is_wsl_linux() -> bool:
    if sys.platform != "linux":
        return False
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def _venv_root() -> Path:
    if os.environ.get("VIRTUAL_ENV"):
        return Path(os.environ["VIRTUAL_ENV"]).expanduser().absolute()
    return Path(sys.executable).expanduser().absolute().parents[1]


def _python_site_packages_dir(venv_root: Path) -> Path:
    return (
        venv_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _nvidia_lib_dirs(site_packages_dir: Path) -> list[Path]:
    nvidia_root = site_packages_dir / "nvidia"
    if not nvidia_root.exists():
        return []
    return sorted(path for path in nvidia_root.glob("*/lib") if path.is_dir())


def _prepend_env_paths(
    environment: dict[str, str],
    variable_name: str,
    paths: list[Path],
) -> bool:
    existing = [
        entry
        for entry in environment.get(variable_name, "").split(":")
        if entry.strip()
    ]
    additions = [str(path) for path in paths if str(path) not in existing]
    if not additions:
        return False
    environment[variable_name] = ":".join([*additions, *existing])
    return True


def ensure_tensorflow_wsl_gpu_env() -> None:
    """Re-exec the current process with the CUDA loader paths expected by WSL2."""

    if not _is_wsl_linux():
        return
    if os.environ.get(_GPU_ENV_BOOTSTRAPPED) == "1":
        return

    venv_root = _venv_root()
    site_packages_dir = _python_site_packages_dir(venv_root)
    nvidia_lib_dirs = _nvidia_lib_dirs(site_packages_dir)
    if not nvidia_lib_dirs:
        return

    updated_env = dict(os.environ)
    changed = False
    wsl_driver_dir = Path("/usr/lib/wsl/lib")

    if wsl_driver_dir.exists():
        changed = _prepend_env_paths(updated_env, "PATH", [wsl_driver_dir]) or changed
        changed = (
            _prepend_env_paths(updated_env, "LD_LIBRARY_PATH", [wsl_driver_dir])
            or changed
        )

    changed = (
        _prepend_env_paths(updated_env, "LD_LIBRARY_PATH", nvidia_lib_dirs) or changed
    )

    if not changed:
        return

    updated_env[_GPU_ENV_BOOTSTRAPPED] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], updated_env)
