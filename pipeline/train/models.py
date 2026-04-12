from __future__ import annotations

from typing import Callable


ModelBuilder = Callable[[tuple[int, ...], int, str], object]


def _compile_model(model: object, loss: str) -> object:
    from keras.metrics import AUC, Precision, Recall, TopKCategoricalAccuracy
    from keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
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


def build_model(input_shape: tuple[int, ...], num_classes: int, loss: str = "categorical_crossentropy") -> object:
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
            Dropout(0.25),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.4),
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return _compile_model(model, loss)


def _build_application_model(
    application_cls: Callable[..., object],
    input_shape: tuple[int, ...],
    num_classes: int,
    loss: str,
    *,
    dense_units: int,
    dropout_rate: float,
) -> object:
    from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from keras.models import Model

    base_model = application_cls(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    if dense_units:
        x = Dense(dense_units, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return _compile_model(model, loss)


def build_resnet_model(input_shape: tuple[int, ...], num_classes: int, loss: str = "categorical_crossentropy") -> object:
    from keras.applications import ResNet50

    return _build_application_model(ResNet50, input_shape, num_classes, loss, dense_units=256, dropout_rate=0.5)


def build_densenet_model(input_shape: tuple[int, ...], num_classes: int, loss: str = "categorical_crossentropy") -> object:
    from keras.applications import DenseNet121

    return _build_application_model(DenseNet121, input_shape, num_classes, loss, dense_units=128, dropout_rate=0.5)


def build_efficientnet_model(input_shape: tuple[int, ...], num_classes: int, loss: str = "categorical_crossentropy") -> object:
    from keras.applications import EfficientNetB3

    return _build_application_model(EfficientNetB3, input_shape, num_classes, loss, dense_units=128, dropout_rate=0.5)


def build_mobilenetv3_model(input_shape: tuple[int, ...], num_classes: int, loss: str = "categorical_crossentropy") -> object:
    from keras.applications import MobileNetV3Large

    return _build_application_model(MobileNetV3Large, input_shape, num_classes, loss, dense_units=128, dropout_rate=0.3)


def build_inception_model(input_shape: tuple[int, ...], num_classes: int, loss: str = "categorical_crossentropy") -> object:
    from keras.applications import InceptionV3

    return _build_application_model(InceptionV3, input_shape, num_classes, loss, dense_units=256, dropout_rate=0.5)


def build_nasnet_model(input_shape: tuple[int, ...], num_classes: int, loss: str = "categorical_crossentropy") -> object:
    from keras.applications import NASNetMobile

    return _build_application_model(NASNetMobile, input_shape, num_classes, loss, dense_units=128, dropout_rate=0.4)


def build_bcnet_model(input_shape: tuple[int, ...], num_classes: int, loss: str = "categorical_crossentropy") -> object:
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
            Dropout(0.3),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.4),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return _compile_model(model, loss)


def build_chexnet_model(input_shape: tuple[int, ...], num_classes: int, loss: str = "categorical_crossentropy") -> object:
    from keras.applications import DenseNet121

    return _build_application_model(DenseNet121, input_shape, num_classes, loss, dense_units=0, dropout_rate=0.5)


def build_vgg19_model(input_shape: tuple[int, ...], num_classes: int, loss: str = "categorical_crossentropy") -> object:
    from keras.applications import VGG19

    return _build_application_model(VGG19, input_shape, num_classes, loss, dense_units=128, dropout_rate=0.5)


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
