"""Inference interface for emotion labels and class probabilities."""

from typing import Sequence

from src.config import EMOTION_LABELS, FEATURE_NAMES


def validate_feature_vector(features: Sequence[float]) -> None:
    """Ensure an inference vector matches the training feature schema."""
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features, received {len(features)}.")


def expected_labels() -> tuple[str, ...]:
    """Return labels in the stable probability-output order."""
    return EMOTION_LABELS
