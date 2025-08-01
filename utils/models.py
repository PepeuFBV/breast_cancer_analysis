from keras.models import Sequential
from keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Activation
)
from keras.metrics import AUC, Precision, Recall, TopKCategoricalAccuracy
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Flatten
from keras.applications import ResNet50, DenseNet121, EfficientNetB3, MobileNetV3Large, InceptionV3, NASNetMobile, VGG19


def build_model(input_shape, num_classes, loss='categorical_crossentropy'):
    """
    Build a Convolutional Neural Network model.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        loss (str): Loss function to use. Default is 'categorical_crossentropy'.
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
        loss=loss,
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy")
        ]
    )
    
    return model


def build_resnet_model(input_shape, num_classes, loss='categorical_crossentropy'):
    """
    Build a ResNet-based model for image classification.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        loss (str): Loss function to use. Default is 'categorical_crossentropy'.
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
    
    model.compile(
        optimizer=Adam(1e-4),
        loss=loss,
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy")
        ]
    )
    
    return model


def build_densenet_model(input_shape, num_classes, loss='categorical_crossentropy'):
    """
    Build a DenseNet-based model for image classification.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        loss (str): Loss function to use. Default is 'categorical_crossentropy'.
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
    
    model.compile(
        optimizer=Adam(1e-4),
        loss=loss,
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy")
        ]
    )
    
    return model


def build_efficientnet_model(input_shape, num_classes, loss='categorical_crossentropy'):
    """
    Build an EfficientNet-based model for image classification.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        loss (str): Loss function to use. Default is 'categorical_crossentropy'.
    Returns:
        keras.models.Model: Compiled EfficientNet model.
    """
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(1e-4),
        loss=loss,
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy")
        ]
    )
    
    return model


def build_mobilenetv3_model(input_shape, num_classes, loss='categorical_crossentropy'):
    """
    Build a MobileNetV3-based model for image classification.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        loss (str): Loss function to use. Default is 'categorical_crossentropy'.
    Returns:
        keras.models.Model: Compiled MobileNetV3 model.
    """
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(1e-4),
        loss=loss,
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy")
        ]
    )
    
    return model


def build_inception_model(input_shape, num_classes, loss='categorical_crossentropy'):
    """
    Build an Inception-based model for image classification.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        loss (str): Loss function to use. Default is 'categorical_crossentropy'.
    Returns:
        keras.models.Model: Compiled Inception model.
    """
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(1e-4),
        loss=loss,
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy")
        ]
    )
    
    return model


def build_nasnet_model(input_shape, num_classes, loss='categorical_crossentropy'):
    """
    Build a NASNet-based model for image classification.
    Parameters:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        loss (str): Loss function to use. Default is 'categorical_crossentropy'.
    Returns:
        keras.models.Model: Compiled NASNet model.
    """
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(1e-4),
        loss=loss, 
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy")
        ]
    )
    
    return model


def build_bcnet_model(input_shape, num_classes, loss='categorical_crossentropy'):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), padding='same'), BatchNormalization(), Activation('relu'),
        Conv2D(32, (3, 3), padding='same'), BatchNormalization(), Activation('relu'),
        MaxPooling2D((2, 2)), Dropout(0.3),

        Conv2D(64, (3, 3), padding='same'), BatchNormalization(), Activation('relu'),
        Conv2D(64, (3, 3), padding='same'), BatchNormalization(), Activation('relu'),
        MaxPooling2D((2, 2)), Dropout(0.4),

        Conv2D(128, (3, 3), padding='same'), BatchNormalization(), Activation('relu'),
        Conv2D(128, (3, 3), padding='same'), BatchNormalization(), Activation('relu'),
        MaxPooling2D((2, 2)), Dropout(0.5),

        Flatten(),
        Dense(256, activation='relu'), Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(1e-4),
        loss=loss,
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy")
        ]
    )
    
    return model


def build_chexnet_model(input_shape, num_classes, loss='categorical_crossentropy'):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)  # softmax for one-hot

    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(1e-4),
        loss=loss,
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy")
        ]
    )
    
    return model


def build_vgg19_model(input_shape, num_classes, loss='categorical_crossentropy'):
    base_model = VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    base_model.trainable = False  # fine-tune later if needed

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(1e-4),
        loss=loss,
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            TopKCategoricalAccuracy(name="top_k_accuracy")
        ]
    )
    
    return model


MODEL_BUILDERS = { # 8 active models
    "custom cnn": build_model,
    "resnet": build_resnet_model,
    "densenet": build_densenet_model,
    "efficientnet": build_efficientnet_model,
    "mobilenetv3": build_mobilenetv3_model,
    "inception": build_inception_model,
    "nasnet": build_nasnet_model,
    "bcnet": build_bcnet_model,
    "chexnet": build_chexnet_model,
    "vgg19": build_vgg19_model
}
