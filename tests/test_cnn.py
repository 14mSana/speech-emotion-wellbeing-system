"""Unit tests for dependency-light log-Mel CNN data contracts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.config import AudioPreprocessingConfig, EMOTION_LABELS
from src.models.cnn import (
    LogMelConfig,
    apply_spectrogram_normalizer,
    encode_emotion_labels,
    evaluate_saved_cnn_test,
    extract_logmel_spectrogram,
    fit_spectrogram_normalizer,
    predict_logmel_wav,
)


def test_logmel_extraction_has_fixed_finite_shape() -> None:
    """A valid signal produces the fixed log-Mel input contract."""
    sample_rate = 22_050
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    config = LogMelConfig(n_mels=8, n_fft=256, hop_length=64, fmax=8_000.0, frames=20)

    image = extract_logmel_spectrogram(
        0.1 * np.sin(2 * np.pi * 220 * time), sample_rate, AudioPreprocessingConfig(), config
    )

    assert image.shape == (8, 20)
    assert np.isfinite(image).all()


def test_normalizer_is_fit_from_train_values_and_not_refit_for_validation() -> None:
    """Training statistics are applied unchanged to later partitions."""
    train = np.stack((np.zeros((2, 3), dtype=np.float32), np.full((2, 3), 2.0, dtype=np.float32)))
    validation = np.full((1, 2, 3), 10.0, dtype=np.float32)

    normalizer = fit_spectrogram_normalizer(train)
    normalized_validation = apply_spectrogram_normalizer(validation, normalizer)

    assert normalizer.mean == pytest.approx((1.0, 1.0))
    assert normalizer.std == pytest.approx((1.0, 1.0))
    assert normalized_validation[0, 0, 0] == pytest.approx(9.0)


def test_cnn_label_encoding_is_canonical_and_rejects_unknown_labels() -> None:
    """CNN output indices remain aligned to the five canonical labels."""
    assert encode_emotion_labels(np.asarray(["angry", "neutral", "happy"])).tolist() == [4, 0, 2]
    with pytest.raises(ValueError, match="Unsupported CNN labels"):
        encode_emotion_labels(np.asarray([*EMOTION_LABELS, "fearful"]))


def _write_cnn_metadata(path: Path, test_evaluation: object = None) -> None:
    """Write the smallest valid CNN metadata contract for inference tests."""
    config = AudioPreprocessingConfig()
    mel_config = LogMelConfig(n_mels=8, n_fft=256, hop_length=64, fmax=8_000.0, frames=20)
    path.write_text(
        json.dumps(
            {
                "emotion_labels": list(EMOTION_LABELS),
                "preprocessing_config": asdict(config),
                "logmel_config": asdict(mel_config),
                "normalizer": {"mean": [0.0] * 8, "std": [1.0] * 8},
                "test_evaluation": test_evaluation,
            }
        ),
        encoding="utf-8",
    )


def test_cnn_wav_prediction_uses_canonical_probability_order(tmp_path: Path) -> None:
    """A CNN-like predictor returns five canonical probabilities for a valid WAV."""
    metadata_path = tmp_path / "cnn_metadata.json"
    wav_path = tmp_path / "tone.wav"
    _write_cnn_metadata(metadata_path)
    sample_rate = 22_050
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sf.write(wav_path, 0.1 * np.sin(2 * np.pi * 220 * time), sample_rate)

    class FixedModel:
        """Minimal Keras-like predictor without TensorFlow as a test dependency."""

        def predict(self, values: np.ndarray, verbose: int = 0) -> np.ndarray:
            assert values.shape == (1, 8, 20, 1)
            return np.asarray([[0.1, 0.2, 0.3, 0.15, 0.25]], dtype=np.float64)

    label, probabilities = predict_logmel_wav(FixedModel(), wav_path, metadata_path)

    assert tuple(probabilities) == EMOTION_LABELS
    assert label == "happy"
    assert probabilities["happy"] == pytest.approx(0.3)


def test_cnn_test_evaluation_cannot_run_twice(tmp_path: Path) -> None:
    """Recorded artifacts reject a second test evaluation before loading test data."""
    metadata_path = tmp_path / "cnn_metadata.json"
    _write_cnn_metadata(metadata_path, test_evaluation={"accuracy": 0.5})

    with pytest.raises(RuntimeError, match="already has a recorded test evaluation"):
        evaluate_saved_cnn_test(
            feature_path=tmp_path / "not-read.csv",
            dataset_root=tmp_path / "not-read-raw",
            model_path=tmp_path / "not-read.keras",
            metadata_path=metadata_path,
        )
