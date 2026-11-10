"""Validation-selected 40-feature classical-model experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import AudioPreprocessingConfig, EMOTION_LABELS, EXTENDED_FEATURE_NAMES, FEATURE_VERSION_V2, MODEL_DIR, PROCESSED_DATA_DIR
from src.data.ravdess import EXPECTED_SPLIT_COUNTS, SPLIT_ACTORS
from src.data.ravdess_v2 import V2_FEATURE_FILENAME, V2_METADATA_FILENAME
from src.models.evaluation import EvaluationResult, evaluate_predictions


RANDOM_SEED = 42
LOGISTIC_MODEL_FILENAME = "ravdess_v2_logistic_regression.joblib"
LOGISTIC_METADATA_FILENAME = "ravdess_v2_logistic_regression_metadata.json"
XGBOOST_MODEL_FILENAME = "ravdess_v2_xgboost.joblib"
XGBOOST_METADATA_FILENAME = "ravdess_v2_xgboost_metadata.json"
RANDOM_FOREST_MODEL_FILENAME = "ravdess_v2_random_forest.joblib"
RANDOM_FOREST_METADATA_FILENAME = "ravdess_v2_random_forest_metadata.json"
RBF_SVM_MODEL_FILENAME = "ravdess_v2_rbf_svm.joblib"
RBF_SVM_METADATA_FILENAME = "ravdess_v2_rbf_svm_metadata.json"
COMPARISON_FILENAME = "ravdess_v2_classical_comparison.json"


@dataclass(frozen=True)
class ClassicalV2Result:
    """Persisted paths and final metrics for the two validation-selected models."""

    logistic_evaluation: EvaluationResult
    xgboost_evaluation: EvaluationResult
    logistic_parameters: dict[str, object]
    xgboost_parameters: dict[str, object]


@dataclass(frozen=True)
class TreeAndSVMV2Result:
    """Persisted metrics and choices for the v2 RF and RBF-SVM experiments."""

    random_forest_evaluation: EvaluationResult
    rbf_svm_evaluation: EvaluationResult
    random_forest_parameters: dict[str, object]
    rbf_svm_parameters: dict[str, object]


def _load_v2_data(feature_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, AudioPreprocessingConfig]:
    """Load the v2 schema and enforce the same exact actor split contract."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"v2 feature metadata is not valid JSON: {metadata_path}") from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("feature_version") != FEATURE_VERSION_V2
        or metadata.get("feature_names") != list(EXTENDED_FEATURE_NAMES)
    ):
        raise ValueError("v2 feature metadata does not match the required 40-feature schema.")
    raw_config = metadata.get("preprocessing_config")
    if not isinstance(raw_config, dict):
        raise ValueError("v2 feature metadata is missing preprocessing_config.")
    try:
        config = AudioPreprocessingConfig(**raw_config)
    except TypeError as error:
        raise ValueError("v2 feature preprocessing_config is invalid.") from error
    frame = pd.read_csv(feature_path, dtype={"actor_id": str, "split": str, "emotion_label": str})
    required = {"actor_id", "split", "emotion_label", *EXTENDED_FEATURE_NAMES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"v2 feature data is missing required columns: {sorted(missing)}")
    observed_features = [column for column in frame.columns if column in EXTENDED_FEATURE_NAMES]
    if observed_features != list(EXTENDED_FEATURE_NAMES):
        raise ValueError("v2 feature data does not preserve the declared feature order.")
    if set(frame["emotion_label"]) - set(EMOTION_LABELS):
        raise ValueError("v2 feature data contains unsupported labels.")
    actual_counts = {split: int((frame["split"] == split).sum()) for split in EXPECTED_SPLIT_COUNTS}
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected v2 split counts: {actual_counts}.")
    if frame.groupby("actor_id")["split"].nunique().gt(1).any():
        raise ValueError("An actor appears in more than one v2 split.")
    for split_name, expected_actors in SPLIT_ACTORS.items():
        observed = set(frame.loc[frame["split"] == split_name, "actor_id"])
        if observed != set(expected_actors):
            raise ValueError(f"Invalid actor assignments for v2 {split_name} split.")
    matrix = frame.loc[:, EXTENDED_FEATURE_NAMES].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("v2 feature data contains non-finite values.")
    return frame, config


def _split(frame: pd.DataFrame, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a named partition in the fixed v2 feature order."""
    partition = frame.loc[frame["split"] == split_name]
    return partition.loc[:, EXTENDED_FEATURE_NAMES].to_numpy(dtype=np.float64), partition["emotion_label"].to_numpy(dtype=str)


def _select_by_validation(
    candidates: list[tuple[dict[str, object], Any]], x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray
) -> tuple[dict[str, object], Any, EvaluationResult]:
    """Fit candidates on train only and select maximum validation macro F1."""
    scored: list[tuple[dict[str, object], Any, EvaluationResult]] = []
    for parameters, model in candidates:
        model.fit(x_train, y_train)
        scored.append((parameters, model, evaluate_predictions(y_validation, model.predict(x_validation))))
    return max(scored, key=lambda item: item[2].macro_f1)


def _ensure_test_not_previously_recorded(model_dir: Path, metadata_names: tuple[str, ...]) -> None:
    """Reject accidental repeat held-out evaluation for named experiments."""
    for metadata_name in metadata_names:
        path = model_dir / metadata_name
        if path.is_file():
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Existing experiment metadata is unreadable: {path}") from error
            if metadata.get("test_evaluation") is not None:
                raise RuntimeError(
                    f"v2 classical test evaluation was already recorded for {metadata_name}; "
                    "refusing to repeat it."
                )


def _write_metadata(
    path: Path, model_type: str, parameters: dict[str, object], config: AudioPreprocessingConfig,
    validation: EvaluationResult, test: EvaluationResult, validation_scores: dict[str, float],
) -> None:
    """Persist a complete feature and evaluation contract beside one model."""
    path.write_text(
        json.dumps(
            {
                "model_type": model_type,
                "feature_version": FEATURE_VERSION_V2,
                "feature_names": list(EXTENDED_FEATURE_NAMES),
                "preprocessing_config": asdict(config),
                "emotion_labels": list(EMOTION_LABELS),
                "random_seed": RANDOM_SEED,
                "actor_splits": SPLIT_ACTORS,
                "split_counts": EXPECTED_SPLIT_COUNTS,
                "selected_parameters": parameters,
                "validation_scores": validation_scores,
                "validation_evaluation": validation.to_dict(),
                "test_evaluation": test.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _update_comparison_artifact(model_dir: Path, results: dict[str, EvaluationResult]) -> None:
    """Merge newly recorded evaluations without erasing other v2 experiments."""
    path = model_dir / COMPARISON_FILENAME
    if path.is_file():
        try:
            comparison = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Existing v2 comparison metadata is unreadable: {path}") from error
    else:
        comparison = {"feature_version": FEATURE_VERSION_V2}
    if comparison.get("feature_version") != FEATURE_VERSION_V2:
        raise ValueError("Existing v2 comparison metadata has an unexpected feature version.")
    for name, result in results.items():
        comparison[name] = {"test": result.to_dict()}
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")


def run_classical_v2_experiments(
    feature_path: Path = PROCESSED_DATA_DIR / V2_FEATURE_FILENAME,
    metadata_path: Path = PROCESSED_DATA_DIR / V2_METADATA_FILENAME,
    model_dir: Path = MODEL_DIR,
) -> ClassicalV2Result:
    """Select v2 models on validation actors, then evaluate each chosen model once.

    Logistic Regression's scaler is fit inside its pipeline on training actors
    only. XGBoost uses no fitted feature scaling. Test actors are accessed only
    after both fixed candidate selections are complete.
    """
    _ensure_test_not_previously_recorded(
        model_dir, (LOGISTIC_METADATA_FILENAME, XGBOOST_METADATA_FILENAME)
    )
    frame, config = _load_v2_data(feature_path, metadata_path)
    x_train, y_train = _split(frame, "train")
    x_validation, y_validation = _split(frame, "validation")

    logistic_candidates = [
        ({"C": c, "class_weight": "balanced"}, Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=2_000, random_state=RANDOM_SEED))]))
        for c in (0.1, 1.0, 10.0)
    ]
    xgb_candidates = [
        (
            {"n_estimators": 200, "max_depth": depth, "learning_rate": learning_rate, "subsample": 0.8, "colsample_bytree": 0.8},
            XGBClassifier(
                objective="multi:softprob", num_class=len(EMOTION_LABELS), eval_metric="mlogloss",
                n_estimators=200, max_depth=depth, learning_rate=learning_rate, subsample=0.8,
                colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=1, tree_method="hist",
            ),
        )
        for depth, learning_rate in ((3, 0.05), (4, 0.05), (3, 0.1))
    ]
    logistic_parameters, logistic_model, logistic_validation = _select_by_validation(
        logistic_candidates, x_train, y_train, x_validation, y_validation
    )
    label_to_index = {label: index for index, label in enumerate(EMOTION_LABELS)}
    y_train_indices = np.asarray([label_to_index[label] for label in y_train], dtype=np.int64)
    # XGBoost requires numeric class labels; selection still evaluates the
    # resulting canonical string labels on validation actors only.
    xgb_scored: list[tuple[dict[str, object], Any, EvaluationResult]] = []
    for parameters, model in xgb_candidates:
        model.fit(x_train, y_train_indices)
        predicted = np.asarray(EMOTION_LABELS, dtype=str)[model.predict(x_validation).astype(int)]
        xgb_scored.append((parameters, model, evaluate_predictions(y_validation, predicted)))
    xgboost_parameters, xgboost_model, xgboost_validation = max(xgb_scored, key=lambda item: item[2].macro_f1)
    # Do not materialize the held-out partition until both configurations have
    # been selected exclusively from training and validation actors.
    x_test, y_test = _split(frame, "test")
    logistic_test = evaluate_predictions(y_test, logistic_model.predict(x_test))
    xgboost_test_predictions = np.asarray(EMOTION_LABELS, dtype=str)[xgboost_model.predict(x_test).astype(int)]
    xgboost_test = evaluate_predictions(y_test, xgboost_test_predictions)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic_model, model_dir / LOGISTIC_MODEL_FILENAME)
    joblib.dump(xgboost_model, model_dir / XGBOOST_MODEL_FILENAME)
    logistic_scores = {
        str(parameters["C"]): result.macro_f1
        for parameters, _, result in [
            (params, model, evaluate_predictions(y_validation, model.predict(x_validation)))
            for params, model in logistic_candidates
        ]
    }
    xgb_scores = {f"depth={parameters['max_depth']},lr={parameters['learning_rate']}": result.macro_f1 for parameters, _, result in xgb_scored}
    _write_metadata(model_dir / LOGISTIC_METADATA_FILENAME, "logistic_regression_v2", logistic_parameters, config, logistic_validation, logistic_test, logistic_scores)
    _write_metadata(model_dir / XGBOOST_METADATA_FILENAME, "xgboost_v2", xgboost_parameters, config, xgboost_validation, xgboost_test, xgb_scores)
    _update_comparison_artifact(
        model_dir,
        {
            "logistic_regression": logistic_test,
            "xgboost": xgboost_test,
        },
    )
    return ClassicalV2Result(logistic_test, xgboost_test, logistic_parameters, xgboost_parameters)


def run_v2_random_forest_and_rbf_svm_experiments(
    feature_path: Path = PROCESSED_DATA_DIR / V2_FEATURE_FILENAME,
    metadata_path: Path = PROCESSED_DATA_DIR / V2_METADATA_FILENAME,
    model_dir: Path = MODEL_DIR,
) -> TreeAndSVMV2Result:
    """Select v2 Random Forest and RBF-SVM models, then test each exactly once.

    Both model families are fitted only on Actors 01--16. The RBF-SVM's
    ``StandardScaler`` is inside its pipeline, so its fitted means and scales
    likewise come only from training actors. Candidate selection uses Actors
    17--20 by macro F1. Test rows are materialized only after both selections.
    """
    _ensure_test_not_previously_recorded(
        model_dir, (RANDOM_FOREST_METADATA_FILENAME, RBF_SVM_METADATA_FILENAME)
    )
    frame, config = _load_v2_data(feature_path, metadata_path)
    x_train, y_train = _split(frame, "train")
    x_validation, y_validation = _split(frame, "validation")

    forest_candidates: list[tuple[dict[str, object], Any]] = [
        (
            {"n_estimators": 300, "max_features": max_features, "class_weight": "balanced"},
            RandomForestClassifier(
                n_estimators=300,
                max_features=max_features,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        )
        for max_features in ("sqrt", None)
    ]
    svm_candidates: list[tuple[dict[str, object], Any]] = [
        (
            {"kernel": "rbf", "C": c, "gamma": "scale", "class_weight": "balanced", "probability": True},
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            C=c,
                            gamma="scale",
                            class_weight="balanced",
                            probability=True,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
        )
        for c in (0.1, 1.0, 10.0)
    ]
    forest_parameters, forest_model, forest_validation = _select_by_validation(
        forest_candidates, x_train, y_train, x_validation, y_validation
    )
    svm_parameters, svm_model, svm_validation = _select_by_validation(
        svm_candidates, x_train, y_train, x_validation, y_validation
    )

    x_test, y_test = _split(frame, "test")
    forest_test = evaluate_predictions(y_test, forest_model.predict(x_test))
    svm_test = evaluate_predictions(y_test, svm_model.predict(x_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest_model, model_dir / RANDOM_FOREST_MODEL_FILENAME)
    joblib.dump(svm_model, model_dir / RBF_SVM_MODEL_FILENAME)
    forest_scores = {
        str(parameters["max_features"]): result.macro_f1
        for parameters, _, result in [
            (params, model, evaluate_predictions(y_validation, model.predict(x_validation)))
            for params, model in forest_candidates
        ]
    }
    svm_scores = {
        str(parameters["C"]): result.macro_f1
        for parameters, _, result in [
            (params, model, evaluate_predictions(y_validation, model.predict(x_validation)))
            for params, model in svm_candidates
        ]
    }
    _write_metadata(
        model_dir / RANDOM_FOREST_METADATA_FILENAME,
        "random_forest_v2",
        forest_parameters,
        config,
        forest_validation,
        forest_test,
        forest_scores,
    )
    _write_metadata(
        model_dir / RBF_SVM_METADATA_FILENAME,
        "rbf_svm_v2",
        svm_parameters,
        config,
        svm_validation,
        svm_test,
        svm_scores,
    )
    _update_comparison_artifact(
        model_dir,
        {"random_forest": forest_test, "rbf_svm": svm_test},
    )
    return TreeAndSVMV2Result(forest_test, svm_test, forest_parameters, svm_parameters)
