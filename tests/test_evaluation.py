"""Regression tests for canonical evaluation metric ordering."""

import pytest

from src.models.evaluation import evaluate_predictions


def test_evaluation_uses_canonical_confusion_matrix_order_and_macro_f1() -> None:
    """Known predictions produce stable metrics in the configured label order."""
    true_labels = ["neutral", "neutral", "calm", "happy", "sad", "angry"]
    predicted_labels = ["neutral", "happy", "calm", "happy", "angry", "angry"]

    result = evaluate_predictions(true_labels, predicted_labels)

    assert result.accuracy == pytest.approx(4 / 6)
    assert result.macro_f1 == pytest.approx(0.6)
    assert result.confusion_matrix == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]
    assert result.classification_report["neutral"]["f1-score"] == pytest.approx(2 / 3)
