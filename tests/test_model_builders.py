from __future__ import annotations

import pytest

from pipeline.train import models
from pipeline.train.models import (
    MODEL_BUILDERS,
    ModelRuntimeConfig,
    _TransferLearningSpec,
    build_bcnet_model,
    build_chexnet_model,
    build_model,
    build_resnet_model,
)

keras_layers = pytest.importorskip("keras.layers")
keras_models = pytest.importorskip("keras.models")

Conv2D = keras_layers.Conv2D
Dense = keras_layers.Dense
Dropout = keras_layers.Dropout
Flatten = keras_layers.Flatten
GlobalAveragePooling2D = keras_layers.GlobalAveragePooling2D
Model = keras_models.Model


def _fake_application_builder(
    *,
    weights: str,
    include_top: bool,
    input_tensor,
) -> Model:
    assert weights == "imagenet"
    assert include_top is False
    outputs = Conv2D(4, (1, 1), activation="relu", name="fake_backbone_conv")(input_tensor)
    return Model(inputs=input_tensor, outputs=outputs, name="fake_backbone")


def test_build_model_applies_runtime_overrides() -> None:
    model = build_model(
        (32, 32, 1),
        3,
        learning_rate=5e-4,
        runtime=ModelRuntimeConfig(
            dense_units=64,
            dropout_rates=(0.1, 0.2, 0.3, 0.4),
        ),
    )

    dropout_rates = [layer.rate for layer in model.layers if isinstance(layer, Dropout)]
    dense_units = [layer.units for layer in model.layers if isinstance(layer, Dense)]

    assert dense_units == [64, 3]
    assert dropout_rates == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(5e-4)
    assert model.loss == "categorical_crossentropy"


def test_build_bcnet_model_keeps_flatten_head_defaults() -> None:
    model = build_bcnet_model((32, 32, 1), 4)

    dropout_rates = [layer.rate for layer in model.layers if isinstance(layer, Dropout)]
    dense_units = [layer.units for layer in model.layers if isinstance(layer, Dense)]

    assert any(isinstance(layer, Flatten) for layer in model.layers)
    assert not any(isinstance(layer, GlobalAveragePooling2D) for layer in model.layers)
    assert dense_units == [256, 4]
    assert dropout_rates == pytest.approx([0.3, 0.4, 0.5, 0.5])


def test_transfer_learning_builder_freezes_backbone(monkeypatch) -> None:
    monkeypatch.setattr(
        models,
        "_transfer_learning_specs",
        lambda: {
            "resnet": _TransferLearningSpec(
                application_cls=_fake_application_builder,
                dense_units=32,
                dropout_rate=0.2,
            )
        },
    )

    model = build_resnet_model((16, 16, 3), 2)

    dense_units = [layer.units for layer in model.layers if isinstance(layer, Dense)]
    dropout_rates = [layer.rate for layer in model.layers if isinstance(layer, Dropout)]
    backbone_conv = next(layer for layer in model.layers if layer.name == "fake_backbone_conv")

    assert dense_units == [32, 2]
    assert dropout_rates == pytest.approx([0.2, 0.2])
    assert backbone_conv.trainable is False
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(1e-4)


def test_chexnet_skips_hidden_dense_layer(monkeypatch) -> None:
    monkeypatch.setattr(
        models,
        "_transfer_learning_specs",
        lambda: {
            "chexnet": _TransferLearningSpec(
                application_cls=_fake_application_builder,
                dense_units=0,
                dropout_rate=0.5,
            )
        },
    )

    model = build_chexnet_model((16, 16, 3), 5)

    dense_units = [layer.units for layer in model.layers if isinstance(layer, Dense)]
    dropout_layers = [layer for layer in model.layers if isinstance(layer, Dropout)]

    assert dense_units == [5]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].rate == pytest.approx(0.5)


def test_model_builders_registry_keys_stable() -> None:
    assert set(MODEL_BUILDERS) == {
        "custom cnn",
        "resnet",
        "densenet",
        "efficientnet",
        "mobilenetv3",
        "inception",
        "nasnet",
        "bcnet",
        "chexnet",
        "vgg19",
    }
