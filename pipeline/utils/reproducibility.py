from __future__ import annotations

import os
import random

import numpy as np


def _env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _should_enable_tensorflow_determinism() -> bool:
    explicit = _env_flag("BREAST_CANCER_ANALYSIS_ENABLE_TF_DETERMINISM")
    if explicit is not None:
        return explicit

    requested_device = str(os.environ.get("BREAST_CANCER_ANALYSIS_REQUESTED_DEVICE") or "").strip().lower()
    if requested_device == "cpu":
        return True
    if requested_device == "gpu":
        return False

    cuda_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if cuda_visible_devices == "-1":
        return True

    return False


def enforce_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    enable_tf_determinism = _should_enable_tensorflow_determinism()
    os.environ["TF_DETERMINISTIC_OPS"] = "1" if enable_tf_determinism else "0"

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        if enable_tf_determinism:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
    except Exception:
        return
