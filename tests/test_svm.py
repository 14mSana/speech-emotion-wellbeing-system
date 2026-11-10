"""Tests for the independently persisted fixed RBF-SVM experiment."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.config import EMOTION_LABELS, FEATURE_NAMES
from src.models.predict import predict_probabilities
from src.models.svm import SVM_PARAMETERS, train_rbf_svm
from tests.test_ravdess import _training_rows, _write_feature_metadata


def test_svm_trains_on_train_partition_and_persists_prediction_contract(tmp_path: Path) -> None:
    """The experimental artifact has canonical probability outputs and no test metrics."""
    feature_path = tmp_path / "features.csv"
    metadata_path = tmp_path / "ravdess_feature_metadata.json"
    model_dir = tmp_path / "models"
    import pandas as pd

    pd.DataFrame(_training_rows()).to_csv(feature_path, index=False)
    _write_feature_metadata(metadata_path)

    result = train_rbf_svm(feature_path, model_dir, metadata_path)
    saved_metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    model = joblib.load(result.model_path)
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    assert result.model_path.name == "ravdess_rbf_svm.joblib"
    assert saved_metadata["parameters"] == SVM_PARAMETERS
    assert saved_metadata["test_evaluation"] is None
    assert saved_metadata["emotion_labels"] == list(EMOTION_LABELS)
    assert tuple(predict_probabilities(model, features)) == EMOTION_LABELS
