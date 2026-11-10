"""Versioned fixed-schema acoustic feature extraction."""

from __future__ import annotations

import librosa
import numpy as np

from src.audio.preprocessing import preprocess_audio
from src.config import AudioPreprocessingConfig, EXTENDED_FEATURE_NAMES, FEATURE_NAMES, N_MFCC


_HOP_LENGTH = 512
_N_FFT = 2_048
_PITCH_FMIN = 50.0
_PITCH_FMAX = 500.0
_SPECTRAL_ROLLOFF_PERCENT = 0.85
_SPECTRAL_CONTRAST_BANDS = 6


def _finite_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Return finite population statistics, using zero for absent values."""
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0, 0.0
    return float(np.mean(finite_values)), float(np.std(finite_values))


def _validated_samples(
    audio: np.ndarray, sample_rate: int, config: AudioPreprocessingConfig | None
) -> tuple[np.ndarray, AudioPreprocessingConfig]:
    """Apply the shared preprocessing and enforce the fixed analysis frame."""
    preprocessing_config = config or AudioPreprocessingConfig()
    samples = preprocess_audio(audio, sample_rate, preprocessing_config)
    if samples.size < _N_FFT:
        raise ValueError(f"Audio recording must contain at least {_N_FFT} samples.")
    return samples, preprocessing_config


def _baseline_features(samples: np.ndarray, config: AudioPreprocessingConfig) -> list[float]:
    """Return the immutable v1 descriptors from already preprocessed audio."""
    mfcc = librosa.feature.mfcc(
        y=samples, sr=config.sample_rate, n_mfcc=N_MFCC, n_fft=_N_FFT, hop_length=_HOP_LENGTH
    )
    mfcc_means = [_finite_mean_std(row)[0] for row in mfcc]
    mfcc_stds = [_finite_mean_std(row)[1] for row in mfcc]
    f0, _, _ = librosa.pyin(
        samples,
        fmin=_PITCH_FMIN,
        fmax=_PITCH_FMAX,
        sr=config.sample_rate,
        frame_length=_N_FFT,
        hop_length=_HOP_LENGTH,
    )
    pitch_mean, pitch_std = _finite_mean_std(f0)
    rms_mean, rms_std = _finite_mean_std(
        librosa.feature.rms(y=samples, frame_length=_N_FFT, hop_length=_HOP_LENGTH)
    )
    zcr_mean, zcr_std = _finite_mean_std(
        librosa.feature.zero_crossing_rate(samples, frame_length=_N_FFT, hop_length=_HOP_LENGTH)
    )
    return [*mfcc_means, *mfcc_stds, pitch_mean, pitch_std, rms_mean, rms_std, zcr_mean, zcr_std]


def _spectral_features(samples: np.ndarray, config: AudioPreprocessingConfig) -> list[float]:
    """Return v2 spectral mean/std pairs across fixed, aligned analysis frames."""
    spectral_kwargs = {"y": samples, "sr": config.sample_rate, "n_fft": _N_FFT, "hop_length": _HOP_LENGTH}
    centroid = librosa.feature.spectral_centroid(**spectral_kwargs)
    bandwidth = librosa.feature.spectral_bandwidth(**spectral_kwargs)
    rolloff = librosa.feature.spectral_rolloff(**spectral_kwargs, roll_percent=_SPECTRAL_ROLLOFF_PERCENT)
    contrast = librosa.feature.spectral_contrast(**spectral_kwargs, n_bands=_SPECTRAL_CONTRAST_BANDS)
    # Contrast has one row per frequency band. Collapse bands per frame first,
    # then aggregate time frames to retain exactly one mean/std pair.
    contrast_frames = np.mean(contrast, axis=0)
    return [
        *_finite_mean_std(centroid),
        *_finite_mean_std(bandwidth),
        *_finite_mean_std(rolloff),
        *_finite_mean_std(contrast_frames),
    ]


def _as_finite_tuple(values: list[float], expected_names: tuple[str, ...]) -> tuple[float, ...]:
    """Validate the declared schema without silently replacing invalid outputs."""
    features = tuple(np.asarray(values, dtype=np.float64).tolist())
    if len(features) != len(expected_names):
        raise RuntimeError("Extracted features do not match the declared schema.")
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced NaN or infinite values.")
    return features


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    config: AudioPreprocessingConfig | None = None,
) -> tuple[float, ...]:
    """Extract the immutable v1 32-feature acoustic vector.

    The ordering is 13 MFCC means, 13 MFCC standard deviations, then pitch,
    RMS energy, and ZCR mean/std pairs. It remains the exact schema consumed
    by the saved Random Forest and RBF-SVM artifacts.
    """
    samples, preprocessing_config = _validated_samples(audio, sample_rate, config)
    return _as_finite_tuple(_baseline_features(samples, preprocessing_config), FEATURE_NAMES)


def extract_extended_features(
    audio: np.ndarray,
    sample_rate: int,
    config: AudioPreprocessingConfig | None = None,
) -> tuple[float, ...]:
    """Extract opt-in v2 features: v1 followed by four spectral mean/std pairs.

    Centroid, bandwidth, and 85%-energy rolloff are aggregated over frames.
    Six-band spectral contrast is averaged across bands per frame, then its
    frame-level population mean/std are calculated. No dataset-fitted
    normalization is applied, so extraction cannot leak across partitions.
    """
    samples, preprocessing_config = _validated_samples(audio, sample_rate, config)
    return _as_finite_tuple(
        [*_baseline_features(samples, preprocessing_config), *_spectral_features(samples, preprocessing_config)],
        EXTENDED_FEATURE_NAMES,
    )


def feature_count() -> int:
    """Return the immutable v1 feature-vector length."""
    return len(FEATURE_NAMES)


def extended_feature_count() -> int:
    """Return the opt-in v2 acoustic-vector length."""
    return len(EXTENDED_FEATURE_NAMES)
