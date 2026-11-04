"""Tests for stable project configuration contracts."""

from src.config import EMOTION_LABELS, FEATURE_NAMES


def test_supported_emotions_are_stable() -> None:
    """The baseline supports the five agreed emotion labels."""
    assert EMOTION_LABELS == ("neutral", "calm", "happy", "sad", "angry")


def test_feature_schema_has_a_fixed_size_and_order() -> None:
    """Feature vectors must keep their documented 32-column ordering."""
    assert len(FEATURE_NAMES) == 32
    assert FEATURE_NAMES[:2] == ("mfcc_01_mean", "mfcc_02_mean")
    assert FEATURE_NAMES[-6:] == (
        "pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean", "zcr_std",
    )
