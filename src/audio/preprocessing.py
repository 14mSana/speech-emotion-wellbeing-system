"""Shared audio preprocessing for training and inference."""

import librosa
import numpy as np

from src.config import AudioPreprocessingConfig


def preprocess_audio(
    audio: np.ndarray, sample_rate: int, config: AudioPreprocessingConfig
) -> np.ndarray:
    """Convert audio to finite mono samples at the configured sample rate.

    Multi-channel input must use SoundFile's ``(frames, channels)`` layout.
    Silent recordings are retained as silence so feature extraction has a
    deterministic, finite representation; empty and non-finite input is
    rejected.

    Args:
        audio: One-dimensional mono or two-dimensional frame-by-channel samples.
        sample_rate: Positive source sample rate in Hertz.
        config: Shared training/inference preprocessing settings.

    Returns:
        A one-dimensional ``float32`` mono waveform at ``config.sample_rate``.

    Raises:
        TypeError: If configuration or audio types are unsupported.
        ValueError: If input shape, samples, or rates are invalid.
    """
    if not isinstance(config, AudioPreprocessingConfig):
        raise TypeError("config must be an AudioPreprocessingConfig instance.")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if config.sample_rate <= 0:
        raise ValueError("config.sample_rate must be a positive integer.")

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim not in (1, 2):
        raise ValueError("Audio must be a one-dimensional or two-dimensional array.")
    if samples.size == 0 or samples.shape[0] == 0:
        raise ValueError("Audio recording is empty.")
    if not np.isfinite(samples).all():
        raise ValueError("Audio recording contains NaN or infinite samples.")

    if samples.ndim == 2:
        if samples.shape[1] == 0:
            raise ValueError("Audio recording has no channels.")
        samples = np.mean(samples, axis=1, dtype=np.float32)

    if sample_rate != config.sample_rate:
        samples = librosa.resample(
            samples, orig_sr=sample_rate, target_sr=config.sample_rate, res_type="soxr_hq"
        )

    if config.trim_silence and np.any(samples):
        samples, _ = librosa.effects.trim(samples, top_db=config.trim_top_db)
    if samples.size == 0:
        raise ValueError("Audio recording contains no samples after preprocessing.")
    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
