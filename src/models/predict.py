"""Inference helpers for canonical emotion labels and probabilities."""

from dataclasses import fields
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.audio.features import extract_features
from src.audio.loading import load_wav
from src.config import AudioPreprocessingConfig, EMOTION_LABELS, FEATURE_NAMES


def validate_feature_vector(features: Sequence[float]) -> None:
    """Ensure an inference vector matches the training feature schema."""
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features, received {len(features)}.")


def expected_labels() -> tuple[str, ...]:
    """Return labels in the stable probability-output order."""
    return EMOTION_LABELS


def predict_probabilities(model: object, features: Sequence[float]) -> dict[str, float]:
    """Predict class probabilities in the fixed :data:`EMOTION_LABELS` order."""
    validate_feature_vector(features)
    feature_array = np.asarray(features, dtype=np.float64)
    if not np.isfinite(feature_array).all():
        raise ValueError("Feature vector contains NaN or infinite values.")
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("model must provide predict_proba and classes_ attributes.")

    raw_probabilities = np.asarray(model.predict_proba(feature_array.reshape(1, -1)), dtype=np.float64)
    if raw_probabilities.shape[0] != 1:
        raise ValueError("Model returned an unexpected probability shape.")
    class_probabilities = dict(zip((str(label) for label in model.classes_), raw_probabilities[0], strict=True))
    model_labels = set(class_probabilities)
    expected_label_set = set(EMOTION_LABELS)
    if model_labels != expected_label_set:
        missing_labels = sorted(expected_label_set - model_labels)
        unsupported_labels = sorted(model_labels - expected_label_set)
        raise ValueError(
            f"Model classes must exactly match supported labels; "
            f"missing={missing_labels or 'none'}, unsupported={unsupported_labels or 'none'}."
        )
    return {label: float(class_probabilities[label]) for label in EMOTION_LABELS}


def predict_label(model: object, features: Sequence[float]) -> str:
    """Return the highest-probability emotion using the canonical label order."""
    probabilities = predict_probabilities(model, features)
    return max(probabilities, key=probabilities.__getitem__)


def load_saved_preprocessing_config(model_metadata_path: Path) -> AudioPreprocessingConfig:
    """Load the exact validated preprocessing configuration saved with a model."""
    if not model_metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata was not found: {model_metadata_path}")
    try:
        metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Model metadata is not valid JSON: {model_metadata_path}") from error
    if not isinstance(metadata, dict) or metadata.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Model metadata feature_names do not match the canonical feature order.")
    raw_config = metadata.get("preprocessing_config")
    config_fields = {field.name for field in fields(AudioPreprocessingConfig)}
    if not isinstance(raw_config, dict) or set(raw_config) != config_fields:
        raise ValueError("Model metadata preprocessing_config is incomplete or invalid.")
    try:
        config = AudioPreprocessingConfig(**raw_config)
    except TypeError as error:
        raise ValueError("Model metadata preprocessing_config is invalid.") from error
    if (
        not isinstance(config.sample_rate, int)
        or isinstance(config.sample_rate, bool)
        or config.sample_rate <= 0
        or not isinstance(config.mono, bool)
        or not isinstance(config.trim_silence, bool)
        or not isinstance(config.trim_top_db, (int, float))
        or isinstance(config.trim_top_db, bool)
        or config.trim_top_db <= 0
    ):
        raise ValueError("Model metadata preprocessing_config contains invalid values.")
    return config


def predict_wav(
    model: object, audio_path: Path, model_metadata_path: Path
) -> tuple[str, dict[str, float]]:
    """Extract saved-config features from a WAV and return label and probabilities."""
    config = load_saved_preprocessing_config(model_metadata_path)
    audio, sample_rate = load_wav(audio_path)
    features = extract_features(audio, sample_rate, config)
    probabilities = predict_probabilities(model, features)
    return predict_label(model, features), probabilities
