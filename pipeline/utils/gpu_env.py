from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

_GPU_ENV_BOOTSTRAPPED = "BREAST_CANCER_ANALYSIS_GPU_ENV_BOOTSTRAPPED"
_WINDOWS_GPU_ENV_BOOTSTRAPPED = "BREAST_CANCER_ANALYSIS_WINDOWS_GPU_ENV_BOOTSTRAPPED"
_GPU_MEMORY_CONFIGURED = False
_WINDOWS_DLL_DIR_HANDLES: list[object] = []
WINDOWS_NATIVE_TF_GPU_MAX_VERSION = (2, 10)
WINDOWS_NATIVE_CUDA_VERSION = "11.2"
WINDOWS_NATIVE_CUDA_DLL = "cudart64_110.dll"


def _is_wsl_linux() -> bool:
    if sys.platform != "linux":
        return False
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def is_native_windows() -> bool:
    return sys.platform == "win32"


def should_bootstrap_wsl_tensorflow_env() -> bool:
    return _is_wsl_linux()


def _venv_root() -> Path:
    if os.environ.get("VIRTUAL_ENV"):
        return Path(os.environ["VIRTUAL_ENV"]).expanduser()
    return Path(sys.executable).expanduser().parents[1]


def _python_site_packages_dir(venv_root: Path) -> Path:
    return venv_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def _nvidia_lib_dirs(site_packages_dir: Path) -> list[Path]:
    nvidia_root = site_packages_dir / "nvidia"
    if not nvidia_root.exists():
        return []
    library_dirs: set[Path] = set()
    for package_dir in nvidia_root.iterdir():
        if not package_dir.is_dir():
            continue
        for directory_name in ("lib", "lib64"):
            candidate = package_dir / directory_name
            if candidate.is_dir():
                library_dirs.add(candidate)
    return sorted(library_dirs)


def _prepend_env_paths(
    environment: dict[str, str],
    variable_name: str,
    paths: list[Path],
) -> bool:
    existing = [entry for entry in environment.get(variable_name, "").split(os.pathsep) if entry.strip()]
    additions = []
    for path in paths:
        normalized_path = str(path)
        if normalized_path not in existing:
            additions.append(normalized_path)
    if not additions:
        return False
    environment[variable_name] = os.pathsep.join([*additions, *existing])
    return True


def _parse_major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _tensorflow_distribution_version() -> str | None:
    for distribution_name in ("tensorflow", "tensorflow-cpu", "tensorflow-intel"):
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _native_windows_cuda_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    for env_var in ("CUDA_PATH_V11_2", "CUDA_PATH"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.append(Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA") / f"v{WINDOWS_NATIVE_CUDA_VERSION}")

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(normalized)
    return unique_candidates


def _native_windows_cuda_runtime_dirs(cuda_root: Path) -> list[Path]:
    bin_dir = cuda_root / "bin"
    if not (bin_dir.is_dir() and (bin_dir / WINDOWS_NATIVE_CUDA_DLL).exists()):
        return []

    runtime_dirs = [bin_dir]
    libnvvp_dir = cuda_root / "libnvvp"
    if libnvvp_dir.is_dir():
        runtime_dirs.append(libnvvp_dir)
    return runtime_dirs


def _register_windows_dll_directories(paths: list[Path]) -> None:
    if not hasattr(os, "add_dll_directory"):
        return

    for path in paths:
        if not path.is_dir():
            continue
        try:
            _WINDOWS_DLL_DIR_HANDLES.append(os.add_dll_directory(str(path)))
        except (FileNotFoundError, OSError):
            continue


def ensure_native_windows_tensorflow_env() -> None:
    """Prepend the TensorFlow 2.10-compatible CUDA runtime dirs on native Windows."""

    if not is_native_windows():
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return
    if os.environ.get(_WINDOWS_GPU_ENV_BOOTSTRAPPED) == "1":
        return

    tensorflow_version = _tensorflow_distribution_version()
    tensorflow_major_minor = _parse_major_minor(tensorflow_version)
    if tensorflow_major_minor is None or tensorflow_major_minor > WINDOWS_NATIVE_TF_GPU_MAX_VERSION:
        return

    for cuda_root in _native_windows_cuda_root_candidates():
        runtime_dirs = _native_windows_cuda_runtime_dirs(cuda_root)
        if not runtime_dirs:
            continue

        _prepend_env_paths(os.environ, "PATH", runtime_dirs)
        os.environ.setdefault("CUDA_PATH", str(cuda_root))
        os.environ.setdefault("CUDA_PATH_V11_2", str(cuda_root))
        _register_windows_dll_directories(runtime_dirs)
        os.environ[_WINDOWS_GPU_ENV_BOOTSTRAPPED] = "1"
        return


def configure_gpu_memory_growth() -> bool:
    """Configure TensorFlow to use GPU memory growth instead of pre-allocating.

    Returns True if configuration was successful, False otherwise.
    """
    global _GPU_MEMORY_CONFIGURED

    if _GPU_MEMORY_CONFIGURED:
        return True

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return False

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            _GPU_MEMORY_CONFIGURED = True
            return True
    except Exception:
        pass

    return False


def clear_gpu_memory() -> None:
    """Aggressively clear GPU memory and Python garbage."""
    # Clear Keras/TensorFlow session
    try:
        from keras import backend as K

        K.clear_session()
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        # Reset default graph
        try:
            tf.compat.v1.reset_default_graph()
        except Exception:
            pass
    except Exception:
        pass

    # Force garbage collection multiple times
    import gc

    for _ in range(3):
        gc.collect()

    # Try to clear CUDA cache if available
    try:
        import tensorflow as tf

        if hasattr(tf.config.experimental, "reset_memory_stats"):
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                try:
                    tf.config.experimental.reset_memory_stats(gpu)
                except Exception:
                    pass
    except Exception:
        pass


def ensure_tensorflow_wsl_gpu_env() -> None:
    """Re-exec the current process with the CUDA loader paths expected by WSL2."""

    if not should_bootstrap_wsl_tensorflow_env():
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return
    if os.environ.get(_GPU_ENV_BOOTSTRAPPED) == "1":
        return

    venv_root = _venv_root()
    site_packages_dir = _python_site_packages_dir(venv_root)
    nvidia_lib_dirs = _nvidia_lib_dirs(site_packages_dir)

    updated_env = dict(os.environ)
    changed = False
    wsl_driver_dir = Path("/usr/lib/wsl/lib")

    if wsl_driver_dir.exists():
        changed = _prepend_env_paths(updated_env, "PATH", [wsl_driver_dir]) or changed
        changed = _prepend_env_paths(updated_env, "LD_LIBRARY_PATH", [wsl_driver_dir]) or changed

    changed = _prepend_env_paths(updated_env, "LD_LIBRARY_PATH", nvidia_lib_dirs) or changed

    if not changed:
        return

    updated_env[_GPU_ENV_BOOTSTRAPPED] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], updated_env)


def bootstrap_tensorflow_runtime_env() -> None:
    """Bootstrap TensorFlow runtime env only when WSL-specific setup is required."""
    if should_bootstrap_wsl_tensorflow_env():
        ensure_tensorflow_wsl_gpu_env()
        return
    ensure_native_windows_tensorflow_env()
