"""Metrics for the fixed speaker-independent evaluation split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import EMOTION_LABELS


@dataclass(frozen=True)
class EvaluationResult:
    """Serializable classification metrics using the canonical label order."""

    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    classification_report: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return metrics in a JSON-compatible representation."""
        return asdict(self)


def evaluate_predictions(
    true_labels: Sequence[str], predicted_labels: Sequence[str]
) -> EvaluationResult:
    """Calculate accuracy, per-class metrics, macro F1, and confusion matrix."""
    if len(true_labels) == 0:
        raise ValueError("Cannot evaluate an empty label sequence.")
    if len(true_labels) != len(predicted_labels):
        raise ValueError("True and predicted label counts must match.")
    unsupported = set(true_labels) | set(predicted_labels)
    unsupported -= set(EMOTION_LABELS)
    if unsupported:
        raise ValueError(f"Unsupported labels in evaluation: {sorted(unsupported)}")

    labels = list(EMOTION_LABELS)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return EvaluationResult(
        accuracy=float(accuracy_score(true_labels, predicted_labels)),
        macro_f1=float(f1_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(true_labels, predicted_labels, labels=labels).astype(int).tolist(),
        classification_report=report,
    )
