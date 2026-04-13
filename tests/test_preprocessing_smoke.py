from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from pipeline.train.preprocessing import (
    binarize_image,
    clahe_image,
    close_image,
    denoise_image,
    dilate_image,
    erode_image,
    iter_preprocessing_tasks,
    lowpass_filter_image,
    open_image,
)


def _sample_image() -> np.ndarray:
    return np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)


@pytest.mark.parametrize(
    ("transform", "kwargs"),
    [
        (denoise_image, {"kernel_size": (3, 3), "sigma": 0}),
        (binarize_image, {"threshold": 127, "max_value": 255, "method": "fixed"}),
        (lowpass_filter_image, {"kernel_size": (3, 3), "iterations": 1, "method": "mean"}),
        (erode_image, {"kernel_size": (3, 3), "iterations": 1}),
        (dilate_image, {"kernel_size": (3, 3), "iterations": 1}),
        (open_image, {"kernel_size": (3, 3), "iterations": 1}),
        (close_image, {"kernel_size": (3, 3), "iterations": 1}),
        (clahe_image, {"tile_grid_size": (3, 3), "iterations": 1}),
    ],
    ids=[
        "denoise",
        "binarize",
        "lowpass",
        "erode",
        "dilate",
        "open",
        "close",
        "clahe",
    ],
)
def test_single_preprocessing_functions_preserve_shape(transform, kwargs) -> None:
    image = _sample_image()

    result = transform(image, **kwargs)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_combined_preprocessing_task_smoke() -> None:
    image = _sample_image()
    tasks = list(
        iter_preprocessing_tasks(
            selected_ids=["denoise__binarize"],
            include_combinations=True,
            param_grids={
                "denoise": {
                    "kernel_size": [(3, 3)],
                    "sigma": [0],
                },
                "binarize": {
                    "threshold": [127],
                    "max_value": [255],
                    "method": ["fixed"],
                },
            },
        )
    )

    assert len(tasks) == 1

    task = tasks[0]
    result = task.apply(image)

    assert task.preproc_id == "denoise__binarize"
    assert task.is_combined is True
    assert result.shape == image.shape
    assert result.dtype == image.dtype


@pytest.mark.parametrize(
    "transform",
    [
        partial(binarize_image, method="invalid"),
        partial(lowpass_filter_image, method="invalid"),
    ],
    ids=["binarize", "lowpass"],
)
def test_preprocessing_functions_reject_unknown_methods(transform) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        transform(_sample_image())
