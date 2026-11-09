"""Reproducible Random Forest training for the prepared RAVDESS features."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import AudioPreprocessingConfig, EMOTION_LABELS, FEATURE_NAMES, MODEL_DIR, PROCESSED_DATA_DIR
from src.data.ravdess import EXPECTED_SPLIT_COUNTS, RAVDESS_EMOTION_CODES, SPLIT_ACTORS
from src.models.evaluation import EvaluationResult, evaluate_predictions


RANDOM_SEED = 42
FEATURE_METADATA_FILENAME = "ravdess_feature_metadata.json"


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts and metrics from one speaker-independent training run."""

    model_path: Path
    metadata_path: Path
    selected_parameters: dict[str, object]
    validation_scores: dict[str, float]
    validation_evaluation: EvaluationResult
    test_evaluation: EvaluationResult


@dataclass(frozen=True)
class FeatureMetadata:
    """Validated feature-generation contracts required for model training."""

    feature_names: tuple[str, ...]
    preprocessing_config: AudioPreprocessingConfig


def _load_feature_metadata(metadata_path: Path) -> FeatureMetadata:
    """Load and validate the metadata written alongside the feature dataset."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Feature metadata was not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Feature metadata is not valid JSON: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Feature metadata must contain a JSON object.")

    required_keys = {"feature_names", "preprocessing_config", "emotion_mapping", "split_actors", "split_counts"}
    missing_keys = required_keys - set(metadata)
    if missing_keys:
        raise ValueError(f"Feature metadata is missing required fields: {sorted(missing_keys)}")
    if metadata["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Feature metadata feature_names do not match the canonical feature order.")
    if metadata["emotion_mapping"] != RAVDESS_EMOTION_CODES:
        raise ValueError("Feature metadata emotion_mapping does not match the approved RAVDESS mapping.")
    expected_split_actors = {name: list(actor_ids) for name, actor_ids in SPLIT_ACTORS.items()}
    if metadata["split_actors"] != expected_split_actors:
        raise ValueError("Feature metadata actor split assignments do not match the approved split.")
    if metadata["split_counts"] != EXPECTED_SPLIT_COUNTS:
        raise ValueError("Feature metadata split counts do not match the approved split.")

    raw_config = metadata["preprocessing_config"]
    if not isinstance(raw_config, dict):
        raise ValueError("Feature metadata preprocessing_config must be an object.")
    config_fields = {field.name for field in fields(AudioPreprocessingConfig)}
    if set(raw_config) != config_fields:
        raise ValueError("Feature metadata preprocessing_config is incomplete or has unknown fields.")
    if not isinstance(raw_config["sample_rate"], int) or isinstance(raw_config["sample_rate"], bool) or raw_config["sample_rate"] <= 0:
        raise ValueError("Feature metadata sample_rate must be a positive integer.")
    if not isinstance(raw_config["mono"], bool) or not isinstance(raw_config["trim_silence"], bool):
        raise ValueError("Feature metadata mono and trim_silence must be boolean values.")
    if (
        not isinstance(raw_config["trim_top_db"], (int, float))
        or isinstance(raw_config["trim_top_db"], bool)
        or raw_config["trim_top_db"] <= 0
    ):
        raise ValueError("Feature metadata trim_top_db must be a positive number.")
    return FeatureMetadata(
        feature_names=tuple(metadata["feature_names"]),
        preprocessing_config=AudioPreprocessingConfig(**raw_config),
    )


def _load_feature_data(feature_path: Path) -> pd.DataFrame:
    """Load and validate the locally prepared, canonical feature dataset."""
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature dataset was not found: {feature_path}")
    frame = pd.read_csv(feature_path, dtype={"actor_id": str, "split": str, "emotion_label": str})
    required_columns = {"split", "actor_id", "emotion_label", *FEATURE_NAMES}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {sorted(missing)}")
    feature_columns = [column for column in frame.columns if column in FEATURE_NAMES]
    if feature_columns != list(FEATURE_NAMES):
        raise ValueError("Feature dataset does not preserve the canonical feature order.")
    if set(frame["emotion_label"]) - set(EMOTION_LABELS):
        raise ValueError("Feature dataset contains unsupported emotion labels.")
    if set(frame["split"]) != set(EXPECTED_SPLIT_COUNTS):
        raise ValueError("Feature dataset does not contain all required split names.")
    actual_counts = {name: int((frame["split"] == name).sum()) for name in EXPECTED_SPLIT_COUNTS}
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected split counts: {actual_counts}; expected {EXPECTED_SPLIT_COUNTS}.")
    if frame.groupby("actor_id")["split"].nunique().gt(1).any():
        raise ValueError("An actor appears in more than one split.")
    for split_name, expected_actors in SPLIT_ACTORS.items():
        observed_actors = set(frame.loc[frame["split"] == split_name, "actor_id"])
        expected_actor_set = set(expected_actors)
        if observed_actors != expected_actor_set:
            missing_actors = sorted(expected_actor_set - observed_actors)
            unexpected_actors = sorted(observed_actors - expected_actor_set)
            raise ValueError(
                f"Invalid actors for {split_name} split; "
                f"missing={missing_actors or 'none'}, unexpected={unexpected_actors or 'none'}."
            )
    matrix = frame.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature dataset contains non-finite feature values.")
    return frame


def _split_xy(frame: pd.DataFrame, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical features and labels for one named partition."""
    partition = frame.loc[frame["split"] == split_name]
    return (
        partition.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64),
        partition["emotion_label"].to_numpy(dtype=str),
    )


def train_random_forest(
    feature_path: Path = PROCESSED_DATA_DIR / "ravdess_features.csv",
    model_dir: Path = MODEL_DIR,
    feature_metadata_path: Path | None = None,
) -> TrainingResult:
    """Select a Random Forest configuration on validation data and test once.

    Candidate estimators fit only training actors. The validation partition
    selects ``max_features`` by macro F1; test metrics are computed only after
    that fixed choice has been made.
    """
    metadata_path = feature_metadata_path or feature_path.parent / FEATURE_METADATA_FILENAME
    feature_metadata = _load_feature_metadata(metadata_path)
    frame = _load_feature_data(feature_path)
    x_train, y_train = _split_xy(frame, "train")
    x_validation, y_validation = _split_xy(frame, "validation")
    x_test, y_test = _split_xy(frame, "test")

    candidate_parameters = (
        {"n_estimators": 300, "max_features": "sqrt"},
        {"n_estimators": 300, "max_features": None},
    )
    candidates: list[tuple[dict[str, object], RandomForestClassifier, EvaluationResult]] = []
    for parameters in candidate_parameters:
        model = RandomForestClassifier(
            **parameters,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)
        validation_result = evaluate_predictions(y_validation, model.predict(x_validation))
        candidates.append((parameters, model, validation_result))

    selected_parameters, selected_model, validation_evaluation = max(
        candidates, key=lambda candidate: candidate[2].macro_f1
    )
    test_evaluation = evaluate_predictions(y_test, selected_model.predict(x_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ravdess_random_forest.joblib"
    model_metadata_path = model_dir / "ravdess_random_forest_metadata.json"
    joblib.dump(selected_model, model_path)
    validation_scores = {
        str(parameters["max_features"]): evaluation.macro_f1
        for parameters, _, evaluation in candidates
    }
    model_metadata_path.write_text(
        json.dumps(
            {
                "feature_names": list(feature_metadata.feature_names),
                "preprocessing_config": asdict(feature_metadata.preprocessing_config),
                "emotion_labels": list(EMOTION_LABELS),
                "random_seed": RANDOM_SEED,
                "actor_splits": SPLIT_ACTORS,
                "split_counts": EXPECTED_SPLIT_COUNTS,
                "selected_parameters": selected_parameters,
                "validation_scores": validation_scores,
                "validation_evaluation": validation_evaluation.to_dict(),
                "test_evaluation": test_evaluation.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrainingResult(
        model_path=model_path,
        metadata_path=model_metadata_path,
        selected_parameters=selected_parameters,
        validation_scores=validation_scores,
        validation_evaluation=validation_evaluation,
        test_evaluation=test_evaluation,
    )


if __name__ == "__main__":
    result = train_random_forest()
    print(f"Saved model: {result.model_path}")
    print(f"Selected parameters: {result.selected_parameters}")
    print(f"Validation macro F1: {result.validation_evaluation.macro_f1:.4f}")
    print(f"Test accuracy: {result.test_evaluation.accuracy:.4f}")
    print(f"Test macro F1: {result.test_evaluation.macro_f1:.4f}")
