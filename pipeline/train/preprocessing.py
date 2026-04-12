from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from pipeline.utils.naming import param_dict_to_display, param_dict_to_file_id, param_dict_to_json


ImageTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PreprocessingDefinition:
    func: Callable[..., np.ndarray]
    params: dict[str, list[Any]]


@dataclass(frozen=True)
class PreprocessingTask:
    preproc_id: str
    params: dict[str, Any]
    param_display: str
    param_id: str
    param_json: str
    is_combined: bool
    apply: ImageTransform


def denoise_image(image: np.ndarray, kernel_size: tuple[int, int] = (5, 5), sigma: float = 0) -> np.ndarray:
    return cv2.GaussianBlur(image, kernel_size, sigma)


def binarize_image(
    image: np.ndarray,
    threshold: int = 127,
    max_value: int = 255,
    method: str = "fixed",
) -> np.ndarray:
    if method == "fixed":
        _, binary = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
        return binary
    if method == "adaptive_mean":
        return cv2.adaptiveThreshold(
            image,
            max_value,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
    if method == "adaptive_gaussian":
        return cv2.adaptiveThreshold(
            image,
            max_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
    raise ValueError("Unknown method: choose 'fixed', 'adaptive_mean', or 'adaptive_gaussian'")


def lowpass_filter_image(
    image: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    iterations: int = 1,
    method: str = "mean",
) -> np.ndarray:
    filtered = image.copy()
    for _ in range(iterations):
        if method == "mean":
            filtered = cv2.blur(filtered, kernel_size)
        elif method == "gaussian":
            filtered = cv2.GaussianBlur(filtered, kernel_size, 0)
        elif method == "median":
            filtered = cv2.medianBlur(filtered, kernel_size[0])
        else:
            raise ValueError("Unknown method: choose 'mean', 'gaussian', or 'median'")
    return filtered


def erode_image(image: np.ndarray, kernel_size: tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)


def dilate_image(image: np.ndarray, kernel_size: tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


def open_image(image: np.ndarray, kernel_size: tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def close_image(image: np.ndarray, kernel_size: tuple[int, int] = (3, 3), iterations: int = 1) -> np.ndarray:
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def clahe_image(
    image: np.ndarray,
    tile_grid_size: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    clahe = cv2.createCLAHE(tileGridSize=tile_grid_size)
    result = image.copy()
    for _ in range(iterations):
        result = clahe.apply(result)
    return result


SINGLE_PREPROCESSING_METHODS: dict[str, PreprocessingDefinition] = {
    "none": PreprocessingDefinition(func=lambda image: image, params={}),
    "denoise": PreprocessingDefinition(
        func=denoise_image,
        params={
            "kernel_size": [(3, 3), (5, 5), (9, 9), (11, 11), (15, 15)],
            "sigma": [0, 1, 3, 5, 7],
        },
    ),
    "binarize": PreprocessingDefinition(
        func=binarize_image,
        params={
            "threshold": [70, 100, 127, 200, 220, 255],
            "max_value": [70, 100, 127, 200, 220, 255],
            "method": ["fixed", "adaptive_mean", "adaptive_gaussian"],
        },
    ),
    "lowpass": PreprocessingDefinition(
        func=lowpass_filter_image,
        params={
            "kernel_size": [(3, 3), (5, 5), (9, 9), (11, 11), (15, 15)],
            "method": ["mean", "gaussian", "median"],
            "iterations": [1, 2, 3, 5, 7],
        },
    ),
    "erode": PreprocessingDefinition(
        func=erode_image,
        params={
            "kernel_size": [(3, 3), (5, 5), (9, 9), (11, 11), (15, 15)],
            "iterations": [1, 2, 3, 5, 7],
        },
    ),
    "dilate": PreprocessingDefinition(
        func=dilate_image,
        params={
            "kernel_size": [(3, 3), (5, 5), (9, 9), (11, 11), (15, 15)],
            "iterations": [1, 2, 3, 5, 7],
        },
    ),
    "open": PreprocessingDefinition(
        func=open_image,
        params={
            "kernel_size": [(3, 3), (5, 5), (9, 9), (11, 11), (15, 15)],
            "iterations": [1, 2, 3, 5, 7],
        },
    ),
    "close": PreprocessingDefinition(
        func=close_image,
        params={
            "kernel_size": [(3, 3), (5, 5), (9, 9), (11, 11), (15, 15)],
            "iterations": [1, 2, 3, 5, 7],
        },
    ),
    "clahe": PreprocessingDefinition(
        func=clahe_image,
        params={
            "tile_grid_size": [(3, 3), (5, 5), (9, 9), (11, 11), (15, 15)],
            "iterations": [1, 2, 3, 5, 7],
        },
    ),
}


def iter_param_grid(param_space: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    if not param_space:
        yield {}
        return

    keys = list(param_space.keys())
    for values in product(*(param_space[key] for key in keys)):
        yield dict(zip(keys, values))


def _build_definitions(
    param_grids: dict[str, dict[str, list[Any]]] | None = None,
) -> dict[str, PreprocessingDefinition]:
    definitions: dict[str, PreprocessingDefinition] = {}
    for preproc_id, definition in SINGLE_PREPROCESSING_METHODS.items():
        definitions[preproc_id] = PreprocessingDefinition(
            func=definition.func,
            params=definition.params if param_grids is None else param_grids.get(preproc_id, {}),
        )
    return definitions


def _build_single_task(
    preproc_id: str,
    params: dict[str, Any],
    definitions: dict[str, PreprocessingDefinition],
) -> PreprocessingTask:
    definition = definitions[preproc_id]

    def apply(image: np.ndarray, *, func=definition.func, bound_params=params) -> np.ndarray:
        return func(image, **bound_params)

    return PreprocessingTask(
        preproc_id=preproc_id,
        params=params,
        param_display=param_dict_to_display(params),
        param_id=param_dict_to_file_id(params),
        param_json=param_dict_to_json(params),
        is_combined=False,
        apply=apply,
    )


def _build_combined_task(
    first_id: str,
    second_id: str,
    first_params: dict[str, Any],
    second_params: dict[str, Any],
    definitions: dict[str, PreprocessingDefinition],
) -> PreprocessingTask:
    first_def = definitions[first_id]
    second_def = definitions[second_id]
    params = {
        f"{first_id}_params": first_params,
        f"{second_id}_params": second_params,
    }

    def apply(
        image: np.ndarray,
        *,
        first_func=first_def.func,
        second_func=second_def.func,
        first_bound=first_params,
        second_bound=second_params,
    ) -> np.ndarray:
        return second_func(first_func(image, **first_bound), **second_bound)

    return PreprocessingTask(
        preproc_id=f"{first_id}__{second_id}",
        params=params,
        param_display=param_dict_to_display(params),
        param_id=param_dict_to_file_id(params),
        param_json=param_dict_to_json(params),
        is_combined=True,
        apply=apply,
    )


def iter_preprocessing_tasks(
    selected_ids: list[str] | None = None,
    *,
    include_combinations: bool = True,
    param_grids: dict[str, dict[str, list[Any]]] | None = None,
) -> Iterator[PreprocessingTask]:
    definitions = _build_definitions(param_grids)
    selected = set(selected_ids) if selected_ids else None

    for preproc_id, definition in definitions.items():
        for params in iter_param_grid(definition.params):
            if selected and preproc_id not in selected:
                continue
            yield _build_single_task(preproc_id, params, definitions)

    if not include_combinations:
        return

    combinable_ids = [name for name in definitions if name != "none"]
    for first_id, second_id in permutations(combinable_ids, 2):
        combined_id = f"{first_id}__{second_id}"
        if selected and combined_id not in selected:
            continue
        for first_params in iter_param_grid(definitions[first_id].params):
            for second_params in iter_param_grid(definitions[second_id].params):
                yield _build_combined_task(first_id, second_id, first_params, second_params, definitions)


def build_legacy_preprocessing_methods(
    param_grids: dict[str, dict[str, list[Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    definitions = _build_definitions(param_grids)
    legacy: dict[str, dict[str, Any]] = {}
    for preproc_id, definition in definitions.items():
        legacy[preproc_id] = {
            "func": definition.func,
            "params": definition.params,
        }

    combinable_ids = [name for name in definitions if name != "none"]
    for first_id, second_id in permutations(combinable_ids, 2):
        first_def = definitions[first_id]
        second_def = definitions[second_id]

        def combined_func(
            image: np.ndarray,
            params1: dict[str, Any],
            params2: dict[str, Any],
            *,
            first_func=first_def.func,
            second_func=second_def.func,
        ) -> np.ndarray:
            return second_func(first_func(image, **params1), **params2)

        legacy[f"{first_id}__{second_id}"] = {
            "func": combined_func,
            "params": {
                f"{first_id}_params": list(iter_param_grid(first_def.params)),
                f"{second_id}_params": list(iter_param_grid(second_def.params)),
            },
        }
    return legacy


LEGACY_PREPROCESSING_METHODS = build_legacy_preprocessing_methods()
