from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModelRuntimeConfig:
    input_channels: int = 3
    batch_size: int | None = None
    dense_units: int | None = None
    dropout_rate: float | None = None
    dropout_rates: tuple[float, ...] = ()


ModelBuilder = Callable[..., object]


def _resolve_dropout_rates(
    runtime: ModelRuntimeConfig | None,
    *,
    defaults: tuple[float, ...],
) -> tuple[float, ...]:
    if runtime is None or not runtime.dropout_rates:
        return defaults
    if len(runtime.dropout_rates) != len(defaults):
        raise ValueError(
            f"Expected {len(defaults)} dropout values, got {len(runtime.dropout_rates)}."
        )
    return runtime.dropout_rates


def _resolve_dense_units(runtime: ModelRuntimeConfig | None, *, default: int) -> int:
    if runtime is None or runtime.dense_units is None:
        return default
    return runtime.dense_units


def _resolve_dropout_rate(runtime: ModelRuntimeConfig | None, *, default: float) -> float:
    if runtime is None or runtime.dropout_rate is None:
        return default
    return runtime.dropout_rate


def _compile_model(model: object, loss: str, learning_rate: float) -> object:
    from keras.metrics import AUC, Precision, Recall, TopKCategoricalAccuracy
    from keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy"),
        ],
    )
    return model


def build_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    from keras.layers import (
        Activation,
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        GlobalAveragePooling2D,
        Input,
        MaxPooling2D,
    )
    from keras.models import Sequential

    dropout_rates = _resolve_dropout_rates(runtime, defaults=(0.25, 0.3, 0.4, 0.5))
    dense_units = _resolve_dense_units(runtime, default=128)

    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(32, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rates[0]),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rates[1]),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rates[2]),
            GlobalAveragePooling2D(),
            Dense(dense_units, activation="relu"),
            Dropout(dropout_rates[3]),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return _compile_model(model, loss, learning_rate)


def _build_application_model(
    application_cls: Callable[..., object],
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str,
    learning_rate: float,
    *,
    dense_units: int,
    dropout_rate: float,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from keras.models import Model

    resolved_dense_units = _resolve_dense_units(runtime, default=dense_units)
    resolved_dropout_rate = _resolve_dropout_rate(runtime, default=dropout_rate)

    base_model = application_cls(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(resolved_dropout_rate)(x)
    if resolved_dense_units:
        x = Dense(resolved_dense_units, activation="relu")(x)
        x = Dropout(resolved_dropout_rate)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return _compile_model(model, loss, learning_rate)


def build_resnet_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    from keras.applications import ResNet50

    return _build_application_model(
        ResNet50,
        input_shape,
        num_classes,
        loss,
        learning_rate,
        dense_units=256,
        dropout_rate=0.5,
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
    from keras.applications import DenseNet121

    return _build_application_model(
        DenseNet121,
        input_shape,
        num_classes,
        loss,
        learning_rate,
        dense_units=128,
        dropout_rate=0.5,
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
    from keras.applications import EfficientNetB3

    return _build_application_model(
        EfficientNetB3,
        input_shape,
        num_classes,
        loss,
        learning_rate,
        dense_units=128,
        dropout_rate=0.5,
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
    from keras.applications import MobileNetV3Large

    return _build_application_model(
        MobileNetV3Large,
        input_shape,
        num_classes,
        loss,
        learning_rate,
        dense_units=128,
        dropout_rate=0.3,
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
    from keras.applications import InceptionV3

    return _build_application_model(
        InceptionV3,
        input_shape,
        num_classes,
        loss,
        learning_rate,
        dense_units=256,
        dropout_rate=0.5,
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
    from keras.applications import NASNetMobile

    return _build_application_model(
        NASNetMobile,
        input_shape,
        num_classes,
        loss,
        learning_rate,
        dense_units=128,
        dropout_rate=0.4,
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
    from keras.layers import (
        Activation,
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        Input,
        MaxPooling2D,
    )
    from keras.models import Sequential

    dropout_rates = _resolve_dropout_rates(runtime, defaults=(0.3, 0.4, 0.5, 0.5))
    dense_units = _resolve_dense_units(runtime, default=256)

    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(32, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rates[0]),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rates[1]),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rates[2]),
            Flatten(),
            Dense(dense_units, activation="relu"),
            Dropout(dropout_rates[3]),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return _compile_model(model, loss, learning_rate)


def build_chexnet_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str = "categorical_crossentropy",
    *,
    learning_rate: float = 1e-4,
    runtime: ModelRuntimeConfig | None = None,
) -> object:
    from keras.applications import DenseNet121

    return _build_application_model(
        DenseNet121,
        input_shape,
        num_classes,
        loss,
        learning_rate,
        dense_units=0,
        dropout_rate=0.5,
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
    from keras.applications import VGG19

    return _build_application_model(
        VGG19,
        input_shape,
        num_classes,
        loss,
        learning_rate,
        dense_units=128,
        dropout_rate=0.5,
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
