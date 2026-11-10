"""Tests for the feature-extraction contract."""

from src.audio.features import extended_feature_count, feature_count
from src.config import EXTENDED_FEATURE_NAMES, FEATURE_NAMES


def test_feature_count_matches_canonical_schema() -> None:
    """The extraction module exposes the shared feature-vector size."""
    assert feature_count() == len(FEATURE_NAMES) == 32


def test_extended_feature_count_matches_spectral_schema() -> None:
    """The opt-in spectral schema has eight values beyond the baseline."""
    assert extended_feature_count() == len(EXTENDED_FEATURE_NAMES) == 40
