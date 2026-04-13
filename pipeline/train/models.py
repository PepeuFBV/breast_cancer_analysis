from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol


@dataclass(frozen=True)
class ModelRuntimeConfig:
    """Optional runtime overrides for model builders."""

    input_channels: int = 3
    batch_size: int | None = None
    dense_units: int | None = None
    dropout_rate: float | None = None
    dropout_rates: tuple[float, ...] = ()


class ModelBuilder(Protocol):
    """Callable signature shared by all registered model builders."""

    def __call__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        loss: str = "categorical_crossentropy",
        *,
        learning_rate: float = 1e-4,
        runtime: ModelRuntimeConfig | None = None,
    ) -> object: ...


@dataclass(frozen=True)
class _TransferLearningSpec:
    """Configuration for a frozen pretrained backbone."""

    application_cls: object
    dense_units: int
    dropout_rate: float


def _resolve_dropout_rates(
    runtime: ModelRuntimeConfig | None,
    *,
    defaults: tuple[float, ...],
) -> tuple[float, ...]:
    """Return per-block dropout values, honoring runtime overrides."""

    if runtime is None or not runtime.dropout_rates:
        return defaults
    if len(runtime.dropout_rates) != len(defaults):
        raise ValueError(
            f"Expected {len(defaults)} dropout values, "
            f"got {len(runtime.dropout_rates)}."
        )
    return runtime.dropout_rates


def _resolve_dense_units(runtime: ModelRuntimeConfig | None, *, default: int) -> int:
    """Return the configured dense width or the builder default."""

    if runtime is None or runtime.dense_units is None:
        return default
    return runtime.dense_units


def _resolve_dropout_rate(
    runtime: ModelRuntimeConfig | None, *, default: float
) -> float:
    """Return the configured dropout rate or the builder default."""

    if runtime is None or runtime.dropout_rate is None:
        return default
    return runtime.dropout_rate


def _build_compile_metrics() -> list[object]:
    """Create the shared classification metric set."""

    from keras.metrics import AUC, Precision, Recall, TopKCategoricalAccuracy

    return [
        "accuracy",
        AUC(name="auc"),
        Precision(name="precision"),
        Recall(name="recall"),
        TopKCategoricalAccuracy(name="top_k_accuracy"),
    ]


def _compile_model(model: object, loss: str, learning_rate: float) -> object:
    """Compile a model with the shared optimizer and metric defaults."""

    from keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=_build_compile_metrics(),
    )
    return model


def _build_conv_block(filters: int, dropout_rate: float) -> list[object]:
    """Return the repeated Conv-BN-ReLU block used by custom CNN builders."""

    from keras.layers import (
        Activation,
        BatchNormalization,
        Conv2D,
        Dropout,
        MaxPooling2D,
    )

    return [
        Conv2D(filters, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(filters, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
    ]


def _build_sequential_classifier_tail(
    num_classes: int,
    *,
    dense_units: int,
    dropout_rate: float,
    pooling_layer_cls: object,
) -> list[object]:
    """Create the dense classifier tail for Sequential CNN builders."""

    from keras.layers import Dense, Dropout

    return [
        pooling_layer_cls(),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax"),
    ]


def _build_transfer_learning_classifier(
    encoded_features: object,
    num_classes: int,
    *,
    dense_units: int,
    dropout_rate: float,
) -> object:
    """Build the pooled classifier head for frozen pretrained backbones."""

    from keras.layers import Dense, Dropout, GlobalAveragePooling2D

    x = GlobalAveragePooling2D()(encoded_features)
    x = Dropout(dropout_rate)(x)
    if dense_units:
        x = Dense(dense_units, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
    return Dense(num_classes, activation="softmax")(x)


def _build_custom_cnn(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str,
    learning_rate: float,
    *,
    dense_units: int,
    dropout_rates: tuple[float, ...],
    pooling_layer_cls: object,
) -> object:
    """Build and compile a custom Sequential CNN classifier."""

    from keras.layers import Input
    from keras.models import Sequential

    model_layers: list[object] = [Input(shape=input_shape)]
    for filters, dropout_rate in zip((32, 64, 128), dropout_rates[:3]):
        model_layers.extend(_build_conv_block(filters, dropout_rate))
    model_layers.extend(
        _build_sequential_classifier_tail(
            num_classes,
            dense_units=dense_units,
            dropout_rate=dropout_rates[3],
            pooling_layer_cls=pooling_layer_cls,
        )
    )

    return _compile_model(Sequential(model_layers), loss, learning_rate)


def _build_transfer_learning_model(
    spec: _TransferLearningSpec,
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str,
    learning_rate: float,
    *,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build and compile a frozen transfer learning model."""

    from keras.layers import Input
    from keras.models import Model

    dense_units = _resolve_dense_units(runtime, default=spec.dense_units)
    dropout_rate = _resolve_dropout_rate(runtime, default=spec.dropout_rate)

    base_model = spec.application_cls(
        weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape)
    )
    base_model.trainable = False

    outputs = _build_transfer_learning_classifier(
        base_model.output,
        num_classes,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
    )
    model = Model(inputs=base_model.input, outputs=outputs)
    return _compile_model(model, loss, learning_rate)


@lru_cache(maxsize=1)
def _transfer_learning_specs() -> dict[str, _TransferLearningSpec]:
    """Lazily load transfer learning backbone defaults."""

    from keras.applications import (
        VGG19,
        DenseNet121,
        EfficientNetB3,
        InceptionV3,
        MobileNetV3Large,
        NASNetMobile,
        ResNet50,
    )

    return {
        "resnet": _TransferLearningSpec(
            application_cls=ResNet50, dense_units=256, dropout_rate=0.5
        ),
        "densenet": _TransferLearningSpec(
            application_cls=DenseNet121, dense_units=128, dropout_rate=0.5
        ),
        "efficientnet": _TransferLearningSpec(
            application_cls=EfficientNetB3, dense_units=128, dropout_rate=0.5
        ),
        "mobilenetv3": _TransferLearningSpec(
            application_cls=MobileNetV3Large, dense_units=128, dropout_rate=0.3
        ),
        "inception": _TransferLearningSpec(
            application_cls=InceptionV3, dense_units=256, dropout_rate=0.5
        ),
        "nasnet": _TransferLearningSpec(
            application_cls=NASNetMobile, dense_units=128, dropout_rate=0.4
        ),
        "chexnet": _TransferLearningSpec(
            application_cls=DenseNet121, dense_units=0, dropout_rate=0.5
        ),
        "vgg19": _TransferLearningSpec(
            application_cls=VGG19, dense_units=128, dropout_rate=0.5
        ),
    }


def _build_named_transfer_learning_model(
    name: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str,
    learning_rate: float,
    *,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build a transfer learning model from the shared spec registry."""

    return _build_transfer_learning_model(
        _transfer_learning_specs()[name],
        input_shape,
        num_classes,
        loss,
        learning_rate,
        runtime=runtime,
    )


def build_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build the baseline custom CNN classifier."""

    from keras.layers import GlobalAveragePooling2D

    dropout_rates = _resolve_dropout_rates(runtime, defaults=(0.25, 0.3, 0.4, 0.5))
    dense_units = _resolve_dense_units(runtime, default=128)
    return _build_custom_cnn(
        input_shape,
        num_classes,
        loss,
        learning_rate,
        dense_units=dense_units,
        dropout_rates=dropout_rates,
        pooling_layer_cls=GlobalAveragePooling2D,
    )


def build_resnet_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build a frozen ResNet50 transfer learning classifier."""

    return _build_named_transfer_learning_model(
        "resnet",
        input_shape,
        num_classes,
        loss,
        learning_rate,
        runtime=runtime,
    )


def build_densenet_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build a frozen DenseNet121 transfer learning classifier."""

    return _build_named_transfer_learning_model(
        "densenet",
        input_shape,
        num_classes,
        loss,
        learning_rate,
        runtime=runtime,
    )


def build_efficientnet_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build a frozen EfficientNetB3 transfer learning classifier."""

    return _build_named_transfer_learning_model(
        "efficientnet",
        input_shape,
        num_classes,
        loss,
        learning_rate,
        runtime=runtime,
    )


def build_mobilenetv3_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build a frozen MobileNetV3Large transfer learning classifier."""

    return _build_named_transfer_learning_model(
        "mobilenetv3",
        input_shape,
        num_classes,
        loss,
        learning_rate,
        runtime=runtime,
    )


def build_inception_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build a frozen InceptionV3 transfer learning classifier."""

    return _build_named_transfer_learning_model(
        "inception",
        input_shape,
        num_classes,
        loss,
        learning_rate,
        runtime=runtime,
    )


def build_nasnet_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build a frozen NASNetMobile transfer learning classifier."""

    return _build_named_transfer_learning_model(
        "nasnet",
        input_shape,
        num_classes,
        loss,
        learning_rate,
        runtime=runtime,
    )


def build_bcnet_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build the BCNet-style custom CNN classifier."""

    from keras.layers import Flatten

    dropout_rates = _resolve_dropout_rates(runtime, defaults=(0.3, 0.4, 0.5, 0.5))
    dense_units = _resolve_dense_units(runtime, default=256)
    return _build_custom_cnn(
        input_shape,
        num_classes,
        loss,
        learning_rate,
        dense_units=dense_units,
        dropout_rates=dropout_rates,
        pooling_layer_cls=Flatten,
    )


def build_chexnet_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build a CheXNet-style DenseNet classifier without a hidden dense layer."""

    return _build_named_transfer_learning_model(
        "chexnet",
        input_shape,
        num_classes,
        loss,
        learning_rate,
        runtime=runtime,
    )


def build_vgg19_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    """Build a frozen VGG19 transfer learning classifier."""

    return _build_named_transfer_learning_model(
        "vgg19",
        input_shape,
        num_classes,
        loss,
        learning_rate,
        runtime=runtime,
    )


MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "custom cnn": build_model,
    "resnet": build_resnet_model,
    "densenet": build_densenet_model,
    "efficientnet": build_efficientnet_model,
    "mobilenetv3": build_mobilenetv3_model,
    "inception": build_inception_model,
    "nasnet": build_nasnet_model,
    "bcnet": build_bcnet_model,
    "chexnet": build_chexnet_model,
    "vgg19": build_vgg19_model,
}
