"""Compatibility wrapper around the reusable training preprocessing package."""

from pipeline.train.preprocessing import (
    LEGACY_PREPROCESSING_METHODS as preprocessing_methods,
)
from pipeline.train.preprocessing import (
    binarize_image,
    clahe_image,
    close_image,
    denoise_image,
    dilate_image,
    erode_image,
    lowpass_filter_image,
    open_image,
)

add_all_2_method_combinations = None

__all__ = [
    "add_all_2_method_combinations",
    "binarize_image",
    "clahe_image",
    "close_image",
    "denoise_image",
    "dilate_image",
    "erode_image",
    "lowpass_filter_image",
    "open_image",
    "preprocessing_methods",
]
