from __future__ import annotations

import os
import sys
from pathlib import Path

_GPU_ENV_BOOTSTRAPPED = "BREAST_CANCER_ANALYSIS_GPU_ENV_BOOTSTRAPPED"
_GPU_MEMORY_CONFIGURED = False


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
        if hasattr(tf.config.experimental, 'reset_memory_stats'):
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                try:
                    tf.config.experimental.reset_memory_stats(gpu)
                except Exception:
                    pass
    except Exception:
        pass


def ensure_tensorflow_wsl_gpu_env() -> None:
    """Re-exec the current process with the CUDA loader paths expected by WSL2."""

    if not _is_wsl_linux():
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
