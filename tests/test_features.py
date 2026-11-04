"""Tests for the feature-extraction contract."""

from src.audio.features import feature_count
from src.config import FEATURE_NAMES


def test_feature_count_matches_canonical_schema() -> None:
    """The extraction module exposes the shared feature-vector size."""
    assert feature_count() == len(FEATURE_NAMES) == 32
