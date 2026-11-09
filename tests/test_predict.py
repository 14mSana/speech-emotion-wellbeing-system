"""Tests for the inference input and label contracts."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier

from src.config import AudioPreprocessingConfig, EMOTION_LABELS, FEATURE_NAMES
from src.models import predict
from src.models.predict import expected_labels, predict_label, predict_probabilities, predict_wav, validate_feature_vector


def test_prediction_contract_uses_agreed_labels() -> None:
    """Probability columns will follow the agreed label order."""
    assert expected_labels() == ("neutral", "calm", "happy", "sad", "angry")


def test_prediction_rejects_wrong_feature_length() -> None:
    """Inference cannot proceed with a feature vector from another schema."""
    with pytest.raises(ValueError, match="Expected 32 features"):
        validate_feature_vector([0.0])


def test_prediction_probabilities_follow_canonical_label_order() -> None:
    """Classifier probability columns are reordered to the project label contract."""
    features = np.zeros((5, 32), dtype=float)
    features[:, 0] = np.arange(5)
    labels = np.array(["angry", "calm", "happy", "neutral", "sad"])
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(features, labels)

    probabilities = predict_probabilities(model, features[3])

    assert tuple(probabilities) == expected_labels()
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert predict_label(model, features[3]) == "neutral"


def test_prediction_rejects_model_with_extra_emotion_class() -> None:
    """A model must expose exactly the agreed five output labels."""
    features = np.zeros((6, 32), dtype=float)
    features[:, 0] = np.arange(6)
    labels = np.array(["neutral", "calm", "happy", "sad", "angry", "fearful"])
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(features, labels)

    with pytest.raises(ValueError, match="unsupported=.*fearful"):
        predict_probabilities(model, features[0])


def test_raw_audio_prediction_uses_saved_preprocessing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw-audio inference passes the exact persisted config to feature extraction."""
    custom_config = AudioPreprocessingConfig(sample_rate=16_000, mono=False, trim_silence=False, trim_top_db=25.0)
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "preprocessing_config": asdict(custom_config),
            }
        ),
        encoding="utf-8",
    )
    wav_path = tmp_path / "tone.wav"
    sf.write(wav_path, np.zeros(16_000, dtype=np.float32), 16_000)
    captured: dict[str, AudioPreprocessingConfig] = {}

    def fake_extract_features(
        audio: np.ndarray, sample_rate: int, config: AudioPreprocessingConfig
    ) -> tuple[float, ...]:
        captured["config"] = config
        return tuple(0.0 for _ in FEATURE_NAMES)

    class FixedProbabilityModel:
        """Minimal model stub implementing the prediction contract."""

        classes_ = np.asarray(EMOTION_LABELS)

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.asarray([[0.1, 0.2, 0.3, 0.15, 0.25]])

    monkeypatch.setattr(predict, "extract_features", fake_extract_features)

    label, probabilities = predict_wav(FixedProbabilityModel(), wav_path, metadata_path)

    assert captured["config"] == custom_config
    assert label == "happy"
    assert tuple(probabilities) == EMOTION_LABELS
