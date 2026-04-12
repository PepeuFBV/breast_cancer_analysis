"""Reusable pipeline modules for dataset preparation, training, and evaluation."""

from pipeline.utils.paths import ProjectPaths, build_project_paths

__all__ = [
    "DEFAULT_EXPERIMENT_CONFIG_PATH",
    "ExperimentConfig",
    "ProjectPaths",
    "build_project_paths",
    "load_experiment_config",
]


def load_experiment_config(path=None):
    from pipeline.config import load_experiment_config as _load_experiment_config

    return _load_experiment_config(path)


def __getattr__(name: str):
    if name in {"DEFAULT_EXPERIMENT_CONFIG_PATH", "ExperimentConfig"}:
        from pipeline.config import DEFAULT_EXPERIMENT_CONFIG_PATH, ExperimentConfig

        exports = {
            "DEFAULT_EXPERIMENT_CONFIG_PATH": DEFAULT_EXPERIMENT_CONFIG_PATH,
            "ExperimentConfig": ExperimentConfig,
        }
        return exports[name]
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
