from keras.models import Sequential
from keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Activation
)
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.applications import DenseNet121
from vit_keras import vit


def build_model(input_shape, num_classes):
    """
    Build a Convolutional Neural Network model.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
    Returns:
        keras.models.Sequential: Compiled CNN model.
    """
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_resnet_model(input_shape, num_classes):
    """
    Build a ResNet-based model for image classification.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
    Returns:
        keras.models.Model: Compiled ResNet model.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_densenet_model(input_shape, num_classes):
    """
    Build a DenseNet-based model for image classification.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
    Returns:
        keras.models.Model: Compiled DenseNet model.
    """
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_vit_model(input_shape, num_classes): # TODO: fix this
    """
    Build a Vision Transformer (ViT) model for image classification.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
    Returns:
        keras.models.Model: Compiled ViT model.
    """
    model = vit.vit_b32(
        num_classes=num_classes,
        input_shape=input_shape,
        include_rescaling=True,
        pretrained="imagenet"
    )
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


MODEL_BUILDERS = {
    "custom cnn": build_model,
    "resnet": build_resnet_model,
    "densenet": build_densenet_model,
    # "vit": build_vit_model
}
