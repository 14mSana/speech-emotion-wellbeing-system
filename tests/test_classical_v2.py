"""Tests for validation-selected v2 classical-model experiments."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from src.config import AudioPreprocessingConfig, EXTENDED_FEATURE_NAMES, FEATURE_VERSION_V2
from src.data.ravdess import EXPECTED_SPLIT_COUNTS, SPLIT_ACTORS
from src.models.classical_v2 import (
    LOGISTIC_METADATA_FILENAME,
    RANDOM_FOREST_METADATA_FILENAME,
    RBF_SVM_METADATA_FILENAME,
    XGBOOST_METADATA_FILENAME,
    run_classical_v2_experiments,
    run_v2_random_forest_and_rbf_svm_experiments,
)
from tests.test_ravdess import _training_rows


def _write_v2_inputs(directory: Path) -> tuple[Path, Path]:
    """Create a complete synthetic actor-isolated v2 feature table."""
    frame = pd.DataFrame(_training_rows())
    for index, name in enumerate(EXTENDED_FEATURE_NAMES[32:], start=1):
        frame[name] = frame["mfcc_01_mean"] + index / 100.0
    feature_path = directory / "ravdess_v2_features.csv"
    metadata_path = directory / "ravdess_v2_feature_metadata.json"
    frame.to_csv(feature_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_version": FEATURE_VERSION_V2,
                "feature_names": list(EXTENDED_FEATURE_NAMES),
                "preprocessing_config": asdict(AudioPreprocessingConfig()),
                "split_actors": {name: list(actors) for name, actors in SPLIT_ACTORS.items()},
                "split_counts": EXPECTED_SPLIT_COUNTS,
            }
        ),
        encoding="utf-8",
    )
    return feature_path, metadata_path


def test_v2_classical_models_select_on_validation_and_guard_test_repeat(tmp_path: Path) -> None:
    """Both models persist distinct v2 artifacts after a single test evaluation."""
    feature_path, metadata_path = _write_v2_inputs(tmp_path)
    model_dir = tmp_path / "models"

    result = run_classical_v2_experiments(feature_path, metadata_path, model_dir)
    logistic_metadata = json.loads((model_dir / LOGISTIC_METADATA_FILENAME).read_text(encoding="utf-8"))
    xgboost_metadata = json.loads((model_dir / XGBOOST_METADATA_FILENAME).read_text(encoding="utf-8"))

    assert result.logistic_evaluation.accuracy == pytest.approx(1.0)
    assert 0.0 <= result.xgboost_evaluation.macro_f1 <= 1.0
    assert logistic_metadata["feature_names"] == list(EXTENDED_FEATURE_NAMES)
    assert xgboost_metadata["test_evaluation"] is not None
    logistic_model = joblib.load(model_dir / "ravdess_v2_logistic_regression.joblib")
    training_rows = pd.read_csv(feature_path).query("split == 'train'")
    assert np.allclose(
        logistic_model.named_steps["scaler"].mean_,
        training_rows.loc[:, EXTENDED_FEATURE_NAMES].to_numpy(dtype=float).mean(axis=0),
    )
    with pytest.raises(RuntimeError, match="already recorded"):
        run_classical_v2_experiments(feature_path, metadata_path, model_dir)


def test_v2_random_forest_and_svm_select_on_validation_and_guard_test_repeat(tmp_path: Path) -> None:
    """The separate v2 RF/SVM artifacts use train-only scaling and test once."""
    feature_path, metadata_path = _write_v2_inputs(tmp_path)
    model_dir = tmp_path / "models"

    result = run_v2_random_forest_and_rbf_svm_experiments(
        feature_path, metadata_path, model_dir
    )
    forest_metadata = json.loads(
        (model_dir / RANDOM_FOREST_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    svm_metadata = json.loads(
        (model_dir / RBF_SVM_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    svm_model = joblib.load(model_dir / "ravdess_v2_rbf_svm.joblib")
    training_rows = pd.read_csv(feature_path).query("split == 'train'")

    assert 0.0 <= result.random_forest_evaluation.macro_f1 <= 1.0
    assert 0.0 <= result.rbf_svm_evaluation.macro_f1 <= 1.0
    assert forest_metadata["feature_names"] == list(EXTENDED_FEATURE_NAMES)
    assert svm_metadata["test_evaluation"] is not None
    assert np.allclose(
        svm_model.named_steps["scaler"].mean_,
        training_rows.loc[:, EXTENDED_FEATURE_NAMES].to_numpy(dtype=float).mean(axis=0),
    )
    with pytest.raises(RuntimeError, match="already recorded"):
        run_v2_random_forest_and_rbf_svm_experiments(feature_path, metadata_path, model_dir)
