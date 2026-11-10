"""Tests for WAV loading, shared preprocessing, and feature extraction."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio.features import extract_extended_features, extract_features
from src.audio.loading import load_wav, validate_wav_path
from src.audio.preprocessing import preprocess_audio
from src.config import AudioPreprocessingConfig, EXTENDED_FEATURE_NAMES, FEATURE_NAMES


def _sine_wave(sample_rate: int = 16_000, duration: float = 0.25) -> np.ndarray:
    """Build a deterministic 220 Hz synthetic waveform."""
    time = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    return (0.25 * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)


def test_load_wav_reads_valid_audio(tmp_path: Path) -> None:
    """A valid WAV returns finite samples and its source rate."""
    wav_path = tmp_path / "tone.wav"
    signal = _sine_wave()
    sf.write(wav_path, signal, 16_000)

    audio, sample_rate = load_wav(wav_path)

    assert sample_rate == 16_000
    assert audio.shape == signal.shape
    assert np.isfinite(audio).all()


def test_load_wav_rejects_empty_and_corrupt_files(tmp_path: Path) -> None:
    """Empty and malformed WAV files fail with clear validation errors."""
    empty_wav = tmp_path / "empty.wav"
    sf.write(empty_wav, np.array([], dtype=np.float32), 16_000)
    corrupt_wav = tmp_path / "corrupt.wav"
    corrupt_wav.write_bytes(b"not a wave file")

    with pytest.raises(ValueError, match="empty"):
        load_wav(empty_wav)
    with pytest.raises(ValueError, match="Could not read"):
        load_wav(corrupt_wav)


def test_validate_wav_path_rejects_non_wav_and_missing_files(tmp_path: Path) -> None:
    """Only existing WAV paths are accepted."""
    with pytest.raises(ValueError, match="Only WAV"):
        validate_wav_path(tmp_path / "clip.mp3")
    with pytest.raises(FileNotFoundError):
        validate_wav_path(tmp_path / "missing.wav")


def test_preprocessing_downmixes_and_resamples_stereo_audio() -> None:
    """Stereo audio becomes mono at the configured deterministic rate."""
    signal = _sine_wave()
    stereo = np.column_stack((signal, -signal))

    processed = preprocess_audio(stereo, 16_000, AudioPreprocessingConfig())

    assert processed.ndim == 1
    assert processed.size > 0
    assert np.isfinite(processed).all()


def test_preprocessing_rejects_empty_or_nonfinite_audio() -> None:
    """Invalid recordings cannot enter the common processing path."""
    config = AudioPreprocessingConfig()
    with pytest.raises(ValueError, match="empty"):
        preprocess_audio(np.array([], dtype=np.float32), 16_000, config)
    with pytest.raises(ValueError, match="NaN or infinite"):
        preprocess_audio(np.array([0.0, np.nan]), 16_000, config)


def test_silence_and_sine_features_are_finite_and_schema_sized() -> None:
    """Silent and voiced synthetic recordings map to finite 32-value vectors."""
    config = AudioPreprocessingConfig()
    silence = np.zeros(16_000, dtype=np.float32)
    tone_features = extract_features(_sine_wave(), 16_000, config)
    silence_features = extract_features(silence, 16_000, config)

    assert len(tone_features) == len(FEATURE_NAMES) == 32
    assert len(silence_features) == len(FEATURE_NAMES)
    assert np.isfinite(tone_features).all()
    assert np.isfinite(silence_features).all()
    assert silence_features[26:28] == (0.0, 0.0)


def test_feature_extraction_rejects_invalid_audio() -> None:
    """Feature extraction reports invalid rather than fabricating a vector."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        extract_features(np.array([0.0, np.inf]), 16_000)


def test_feature_extraction_rejects_too_short_audio() -> None:
    """Recordings without one analysis frame are invalid for pitch extraction."""
    with pytest.raises(ValueError, match="at least 2048 samples"):
        extract_features(np.array([0.0, 0.1], dtype=np.float32), 22_050)


def test_extended_features_are_ordered_finite_and_deterministic() -> None:
    """v2 preserves v1 exactly and provides finite spectral statistics."""
    config = AudioPreprocessingConfig(sample_rate=16_000, trim_silence=False)
    tone = _sine_wave()
    silence = np.zeros(16_000, dtype=np.float32)

    first = extract_extended_features(tone, 16_000, config)
    second = extract_extended_features(tone, 16_000, config)
    silent = extract_extended_features(silence, 16_000, config)

    assert len(first) == len(EXTENDED_FEATURE_NAMES) == 40
    assert first[: len(FEATURE_NAMES)] == extract_features(tone, 16_000, config)
    assert first == second
    assert np.isfinite(first).all()
    assert np.isfinite(silent).all()


def test_extended_features_share_invalid_and_short_audio_contract() -> None:
    """v2 reports invalid input instead of emitting fabricated features."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        extract_extended_features(np.array([0.0, np.inf]), 16_000)
    with pytest.raises(ValueError, match="at least 2048 samples"):
        extract_extended_features(np.array([0.0, 0.1], dtype=np.float32), 22_050)
