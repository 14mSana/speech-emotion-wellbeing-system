"""Fixed RBF-SVM experiment artifact for the five-class RAVDESS baseline.

This module intentionally trains on the approved training actors only and
uses validation actors solely for reporting.  It never evaluates test actors;
the separately recorded final test evaluation remains outside this artifact
creation path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import EMOTION_LABELS, MODEL_DIR, PROCESSED_DATA_DIR
from src.data.ravdess import EXPECTED_SPLIT_COUNTS, SPLIT_ACTORS
from src.models.evaluation import EvaluationResult, evaluate_predictions
from src.models.train import (
    FEATURE_METADATA_FILENAME,
    _load_feature_data,
    _load_feature_metadata,
    _split_xy,
)


RANDOM_SEED = 42
SVM_MODEL_FILENAME = "ravdess_rbf_svm.joblib"
SVM_METADATA_FILENAME = "ravdess_rbf_svm_metadata.json"
SVM_PARAMETERS: dict[str, object] = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    # This does not select or tune a model; it exposes the probability API
    # required by the optional comparison interface.
    "probability": True,
}


@dataclass(frozen=True)
class SVMTrainingResult:
    """Paths and validation metrics from the fixed train-only SVM experiment."""

    model_path: Path
    metadata_path: Path
    validation_evaluation: EvaluationResult


def train_rbf_svm(
    feature_path: Path = PROCESSED_DATA_DIR / "ravdess_features.csv",
    model_dir: Path = MODEL_DIR,
    feature_metadata_path: Path | None = None,
) -> SVMTrainingResult:
    """Train the preselected RBF-SVM and report validation metrics only.

    ``StandardScaler`` is fit solely on Actors 01--16.  Actor and schema
    validation is shared with the production training path, including exact
    split membership checks.  The held-out test partition is not extracted or
    evaluated here.
    """
    metadata_path = feature_metadata_path or feature_path.parent / FEATURE_METADATA_FILENAME
    feature_metadata = _load_feature_metadata(metadata_path)
    frame = _load_feature_data(feature_path)
    x_train, y_train = _split_xy(frame, "train")
    x_validation, y_validation = _split_xy(frame, "validation")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(**SVM_PARAMETERS)),
        ]
    )
    model.fit(x_train, y_train)
    validation_evaluation = evaluate_predictions(y_validation, model.predict(x_validation))

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / SVM_MODEL_FILENAME
    model_metadata_path = model_dir / SVM_METADATA_FILENAME
    joblib.dump(model, model_path)
    model_metadata_path.write_text(
        json.dumps(
            {
                "model_type": "rbf_svm_experiment",
                "feature_names": list(feature_metadata.feature_names),
                "preprocessing_config": asdict(feature_metadata.preprocessing_config),
                "emotion_labels": list(EMOTION_LABELS),
                "random_seed": RANDOM_SEED,
                "actor_splits": SPLIT_ACTORS,
                "split_counts": EXPECTED_SPLIT_COUNTS,
                "parameters": SVM_PARAMETERS,
                "validation_evaluation": validation_evaluation.to_dict(),
                "test_evaluation": None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return SVMTrainingResult(model_path, model_metadata_path, validation_evaluation)


if __name__ == "__main__":
    result = train_rbf_svm()
    print(f"Saved experimental SVM: {result.model_path}")
    print(f"Validation accuracy: {result.validation_evaluation.accuracy:.4f}")
    print(f"Validation macro F1: {result.validation_evaluation.macro_f1:.4f}")
