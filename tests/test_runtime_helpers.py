from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.train.models import ModelRuntimeConfig, _resolve_dropout_rates
from pipeline.train.runner import (
    _default_model_runtime,
    _prepare_model_inputs,
    load_split_dataframe,
)


def test_default_model_runtime_uses_family_defaults() -> None:
    assert _default_model_runtime("custom cnn", batch_size=8) == ModelRuntimeConfig(
        input_channels=1,
        batch_size=8,
    )
    assert _default_model_runtime("resnet", batch_size=8) == ModelRuntimeConfig(
        input_channels=3,
        batch_size=4,
    )


def test_prepare_model_inputs_keeps_grayscale_shape_and_runtime_batch_size() -> None:
    images = np.full((2, 4, 4), 255, dtype=np.uint8)

    model_inputs, input_shape, effective_batch_size = _prepare_model_inputs(
        images,
        model_name="custom cnn",
        batch_size=2,
        model_runtime=ModelRuntimeConfig(input_channels=1, batch_size=6),
    )

    assert model_inputs.shape == (2, 4, 4, 1)
    assert input_shape == (4, 4, 1)
    assert effective_batch_size == 6
    assert model_inputs.dtype == np.float32
    assert np.allclose(model_inputs[..., 0], 1.0)


def test_prepare_model_inputs_expands_rgb_channels() -> None:
    images = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)

    model_inputs, input_shape, effective_batch_size = _prepare_model_inputs(
        images,
        model_name="resnet",
        batch_size=2,
        model_runtime=ModelRuntimeConfig(input_channels=3, batch_size=5),
    )

    assert model_inputs.shape == (2, 4, 4, 3)
    assert input_shape == (4, 4, 3)
    assert effective_batch_size == 5
    assert np.allclose(model_inputs[..., 0], model_inputs[..., 1])
    assert np.allclose(model_inputs[..., 1], model_inputs[..., 2])


def test_prepare_model_inputs_rejects_invalid_input_channels() -> None:
    images = np.zeros((1, 4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported input_channels"):
        _prepare_model_inputs(
            images,
            model_name="resnet",
            batch_size=2,
            model_runtime=ModelRuntimeConfig(input_channels=2),
        )


def test_load_split_dataframe_maps_supported_labels(tmp_path) -> None:
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "image_path": ["image_a.png", "image_b.png"],
            "label": ["1", "4A"],
        }
    ).to_csv(split_path, index=False)

    dataframe = load_split_dataframe(split_path)

    assert dataframe["label"].tolist() == [0, 3]


def test_load_split_dataframe_rejects_unknown_labels(tmp_path) -> None:
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "image_path": ["image_a.png"],
            "label": ["7"],
        }
    ).to_csv(split_path, index=False)

    with pytest.raises(ValueError, match="Invalid labels found"):
        load_split_dataframe(split_path)


def test_resolve_dropout_rates_rejects_invalid_length() -> None:
    with pytest.raises(ValueError, match="Expected 4 dropout values"):
        _resolve_dropout_rates(
            ModelRuntimeConfig(dropout_rates=(0.1, 0.2)),
            defaults=(0.25, 0.3, 0.4, 0.5),
        )
