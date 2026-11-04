"""Fixed-schema audio feature extraction interface."""

from typing import Any

from src.config import FEATURE_NAMES


def extract_features(audio: Any, sample_rate: int) -> tuple[float, ...]:
    """Return features in the exact order defined by ``FEATURE_NAMES``."""
    raise NotImplementedError("Feature extraction is not implemented in Phase 1.")


def feature_count() -> int:
    """Return the stable baseline feature-vector length."""
    return len(FEATURE_NAMES)
