"""Compatibility wrapper around the reusable training model registry."""

from pipeline.train.models import (
    MODEL_BUILDERS,
    build_bcnet_model,
    build_chexnet_model,
    build_densenet_model,
    build_efficientnet_model,
    build_inception_model,
    build_mobilenetv3_model,
    build_model,
    build_nasnet_model,
    build_resnet_model,
    build_vgg19_model,
)

__all__ = [
    "MODEL_BUILDERS",
    "build_bcnet_model",
    "build_chexnet_model",
    "build_densenet_model",
    "build_efficientnet_model",
    "build_inception_model",
    "build_mobilenetv3_model",
    "build_model",
    "build_nasnet_model",
    "build_resnet_model",
    "build_vgg19_model",
]
