from __future__ import annotations

VALID_LABELS = ("1", "2", "3", "4a", "4b", "4c", "5", "6")
LABEL_MAPPING = {label: index for index, label in enumerate(VALID_LABELS)}
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_AUGMENTATIONS_PER_IMAGE = 3
DEFAULT_SAMPLES_PER_CLASS = 35
DEFAULT_TEST_SIZE = 0.25
DEFAULT_RANDOM_STATE = 42
