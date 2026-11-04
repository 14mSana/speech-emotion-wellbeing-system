"""Fixed-schema acoustic feature extraction."""

import librosa
import numpy as np

from src.audio.preprocessing import preprocess_audio
from src.config import AudioPreprocessingConfig, FEATURE_NAMES, N_MFCC

_HOP_LENGTH = 512
_PITCH_FMIN = 50.0
_PITCH_FMAX = 500.0


def _finite_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Return finite population statistics, using zero for absent values."""
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0, 0.0
    return float(np.mean(finite_values)), float(np.std(finite_values))


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    config: AudioPreprocessingConfig | None = None,
) -> tuple[float, ...]:
    """Extract the canonical 32 finite acoustic features from a recording.

    The exact ordering is 13 MFCC means, 13 MFCC standard deviations, then
    pitch, RMS energy, and zero-crossing-rate mean/std pairs. Missing voiced
    pitch frames (including fully unvoiced or silent audio) contribute zero
    pitch statistics. The supplied preprocessing configuration is applied here
    to keep extraction identical for training and inference.

    Args:
        audio: Mono or SoundFile-layout multi-channel audio samples.
        sample_rate: Source sample rate in Hertz.
        config: Shared preprocessing configuration; defaults to the project one.

    Returns:
        The 32 values ordered exactly as :data:`src.config.FEATURE_NAMES`.

    Raises:
        ValueError: If audio is invalid or too short to analyse.
    """
    preprocessing_config = config or AudioPreprocessingConfig()
    samples = preprocess_audio(audio, sample_rate, preprocessing_config)
    # Keep a stable analysis frame. Shorter recordings cannot support the
    # configured 50 Hz pitch floor with pyin, so reject them clearly.
    n_fft = 2_048
    if samples.size < n_fft:
        raise ValueError(f"Audio recording must contain at least {n_fft} samples.")
    mfcc = librosa.feature.mfcc(
        y=samples,
        sr=preprocessing_config.sample_rate,
        n_mfcc=N_MFCC,
        n_fft=n_fft,
        hop_length=_HOP_LENGTH,
    )
    mfcc_means = [_finite_mean_std(row)[0] for row in mfcc]
    mfcc_stds = [_finite_mean_std(row)[1] for row in mfcc]

    f0, _, _ = librosa.pyin(
        samples,
        fmin=_PITCH_FMIN,
        fmax=_PITCH_FMAX,
        sr=preprocessing_config.sample_rate,
        frame_length=n_fft,
        hop_length=_HOP_LENGTH,
    )
    pitch_mean, pitch_std = _finite_mean_std(f0)
    rms_mean, rms_std = _finite_mean_std(
        librosa.feature.rms(y=samples, frame_length=n_fft, hop_length=_HOP_LENGTH)
    )
    zcr_mean, zcr_std = _finite_mean_std(
        librosa.feature.zero_crossing_rate(samples, frame_length=n_fft, hop_length=_HOP_LENGTH)
    )

    features = tuple(
        np.nan_to_num(
            np.asarray(
                [*mfcc_means, *mfcc_stds, pitch_mean, pitch_std, rms_mean, rms_std, zcr_mean, zcr_std],
                dtype=np.float64,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).tolist()
    )
    if len(features) != len(FEATURE_NAMES):
        raise RuntimeError("Extracted features do not match the canonical schema.")
    return features


def feature_count() -> int:
    """Return the stable baseline feature-vector length."""
    return len(FEATURE_NAMES)
