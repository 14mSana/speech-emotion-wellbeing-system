"""Tests for the inference input and label contracts."""

import pytest

from src.models.predict import expected_labels, validate_feature_vector


def test_prediction_contract_uses_agreed_labels() -> None:
    """Probability columns will follow the agreed label order."""
    assert expected_labels() == ("neutral", "calm", "happy", "sad", "angry")


def test_prediction_rejects_wrong_feature_length() -> None:
    """Inference cannot proceed with a feature vector from another schema."""
    with pytest.raises(ValueError, match="Expected 32 features"):
        validate_feature_vector([0.0])
