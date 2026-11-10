"""Tests for stable project configuration contracts."""

from src.config import EMOTION_LABELS, EXTENDED_FEATURE_NAMES, FEATURE_NAMES, SPECTRAL_FEATURE_NAMES


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


def test_extended_feature_schema_preserves_v1_then_adds_spectral_properties() -> None:
    """The v2 schema cannot reorder or replace the production v1 schema."""
    assert EXTENDED_FEATURE_NAMES[: len(FEATURE_NAMES)] == FEATURE_NAMES
    assert EXTENDED_FEATURE_NAMES[-len(SPECTRAL_FEATURE_NAMES) :] == SPECTRAL_FEATURE_NAMES
    assert SPECTRAL_FEATURE_NAMES == (
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_contrast_mean", "spectral_contrast_std",
    )
